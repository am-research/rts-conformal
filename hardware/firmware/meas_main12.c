/*
 * meas_main.c -- pWCET measurement harness for NUCLEO-F746ZG.
 *
 * Runs TACLeBench kernels under timed activations and streams the cycle counts
 * as CSV over the ST-LINK virtual COM port, in the format expected by
 * analyze_benchmark_traces.py.
 *
 * Design notes that matter for the paper:
 *
 *  - The core runs at 216 MHz with 7 flash wait states, so the core is far
 *    faster than flash. That gap is what makes cache behaviour visible in the
 *    timing; at the 16 MHz reset clock there are zero wait states and the
 *    caches would have almost nothing to do.
 *
 *  - Each benchmark is run twice: once with the L1 instruction and data caches
 *    enabled, once with both disabled. The cache-disabled arm is a controlled
 *    contrast, not a realistic configuration -- it isolates how much of the
 *    observed tail comes from cache state.
 *
 *  - Two independent sources of variability are applied per activation:
 *      (a) randomised kernel inputs, so the executed path varies;
 *      (b) a randomised cache state, produced by touching a pseudo-random
 *          subset of a scratch buffer larger than the D-cache.
 *    (b) is the software analogue of the randomised cache placement assumed by
 *    MBPTA, and it is what makes successive activations plausibly exchangeable
 *    rather than serially dependent.
 *
 *  - Timing uses DWT_CYCCNT. The counter is 32-bit and wraps roughly every
 *    20 s at 216 MHz, but each activation is far shorter than that, and
 *    unsigned subtraction is correct across a wrap.
 */

#include <stdint.h>

/* ---- register map ----------------------------------------------------- */
#define RCC_BASE      0x40023800u
#define RCC_CR        (*(volatile uint32_t *)(RCC_BASE + 0x00))
#define RCC_PLLCFGR   (*(volatile uint32_t *)(RCC_BASE + 0x04))
#define RCC_CFGR      (*(volatile uint32_t *)(RCC_BASE + 0x08))
#define RCC_AHB1ENR   (*(volatile uint32_t *)(RCC_BASE + 0x30))
#define RCC_APB1ENR   (*(volatile uint32_t *)(RCC_BASE + 0x40))

#define FLASH_ACR     (*(volatile uint32_t *)0x40023C00u)
#define PWR_CR1       (*(volatile uint32_t *)0x40007000u)
#define PWR_CSR1      (*(volatile uint32_t *)0x40007004u)

#define GPIOB_BASE    0x40020400u
#define GPIOD_BASE    0x40020C00u
#define GPIO_MODER(b) (*(volatile uint32_t *)((b) + 0x00))
#define GPIO_OSPEEDR(b)(*(volatile uint32_t *)((b) + 0x08))
#define GPIO_BSRR(b)  (*(volatile uint32_t *)((b) + 0x18))
#define GPIO_AFRH(b)  (*(volatile uint32_t *)((b) + 0x24))

#define USART3_BASE   0x40004800u
#define USART3_CR1    (*(volatile uint32_t *)(USART3_BASE + 0x00))
#define USART3_BRR    (*(volatile uint32_t *)(USART3_BASE + 0x0C))
#define USART3_ISR    (*(volatile uint32_t *)(USART3_BASE + 0x1C))
#define USART3_RDR    (*(volatile uint32_t *)(USART3_BASE + 0x24))
#define USART3_TDR    (*(volatile uint32_t *)(USART3_BASE + 0x28))

#define DEMCR         (*(volatile uint32_t *)0xE000EDFCu)
#define DWT_CTRL      (*(volatile uint32_t *)0xE0001000u)
#define DWT_CYCCNT    (*(volatile uint32_t *)0xE0001004u)
#define DWT_LAR       (*(volatile uint32_t *)0xE0001FB0u)
#define DWT_LAR_KEY   0xC5ACCE55u

#define SCB_CCR       (*(volatile uint32_t *)0xE000ED14u)
#define SCB_CCSIDR    (*(volatile uint32_t *)0xE000ED80u)
#define SCB_CSSELR    (*(volatile uint32_t *)0xE000ED84u)
#define SCB_ICIALLU   (*(volatile uint32_t *)0xE000EF50u)
#define SCB_DCISW     (*(volatile uint32_t *)0xE000EF60u)
#define SCB_DCCISW    (*(volatile uint32_t *)0xE000EF74u)

#define CPACR         (*(volatile uint32_t *)0xE000ED88u)
#define CCR_IC        (1u << 17)
#define CCR_DC        (1u << 16)

#define DSB()  __asm__ volatile("dsb 0xF" ::: "memory")
#define ISB()  __asm__ volatile("isb 0xF" ::: "memory")

#define SYSCLK_HZ     216000000u
#define APB1_HZ       (SYSCLK_HZ / 4u)
#define BAUD          921600u

/* Number of timed activations per (benchmark, cache arm). Raise for the full
   campaign; see README. */
#ifndef N_SAMPLES
#define N_SAMPLES     3000u
#endif
#define N_WARMUP      100u

/* ---- TACLeBench kernels ----------------------------------------------- */
void bsort_init(void);       void bsort_main(void);
void insertsort_init(void);  void insertsort_main(void);
void quicksort_init(void);   void quicksort_main(void);
void ludcmp_init(void);      void ludcmp_main(void);
void fft_init(void);         void fft_main(void);
void ludcmp_init(void);      void ludcmp_main(void);
void fft_init(void);         void fft_main(void);

extern unsigned int insertsort_a[11];
extern char quicksort_strings[681][20];
extern double ludcmp_a[50][50];
extern int fft_input_data[2048];

/* Shared PRNG, also used by the patched kernel initialisers. */
volatile uint32_t tacle_rng_state = 0x12345678u;

uint32_t tacle_rand(void)
{
    /* xorshift32: cheap, and its cost is outside the timed region anyway */
    uint32_t x = tacle_rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    tacle_rng_state = x;
    return x;
}

/* ---- output ----------------------------------------------------------- */
static void uart_putc(char c)
{
    while (!(USART3_ISR & (1u << 7))) { }
    USART3_TDR = (uint32_t)c;
}

static void uart_puts(const char *s)
{
    while (*s) { if (*s == '\n') uart_putc('\r'); uart_putc(*s++); }
}

static void uart_putu32(uint32_t v)
{
    char buf[11]; int i = 10; buf[10] = '\0';
    if (v == 0) { uart_putc('0'); return; }
    while (v && i > 0) { buf[--i] = (char)('0' + (v % 10u)); v /= 10u; }
    uart_puts(&buf[i]);
}

/* ---- clock ------------------------------------------------------------ */
static int clock_to_216mhz(void)
{
    volatile uint32_t t;

    RCC_APB1ENR |= (1u << 28);              /* PWR clock */
    PWR_CR1 |= (3u << 14);                  /* voltage scale 1 */

    /* PLL from HSI(16 MHz): M=8 -> 2 MHz, N=216 -> 432 MHz VCO, P=2 -> 216 MHz.
       HSI is used rather than HSE so the board works regardless of the
       ST-LINK MCO solder-bridge configuration. Cycle counts are unaffected by
       the small HSI frequency tolerance because we count cycles, not seconds. */
    RCC_PLLCFGR = 8u | (216u << 6) | (0u << 16) | (0u << 22) | (9u << 24);

    RCC_CR |= (1u << 24);                   /* PLL on */
    for (t = 0; t < 2000000u; t++) if (RCC_CR & (1u << 25)) break;
    if (!(RCC_CR & (1u << 25))) return 0;

    PWR_CR1 |= (1u << 16);                  /* overdrive enable */
    for (t = 0; t < 2000000u; t++) if (PWR_CSR1 & (1u << 16)) break;
    PWR_CR1 |= (1u << 17);                  /* overdrive switch */
    for (t = 0; t < 2000000u; t++) if (PWR_CSR1 & (1u << 17)) break;

    /* Flash latency must be raised before the clock is. 7 WS at 216 MHz,
       plus ART accelerator and prefetch. */
    FLASH_ACR = 7u | (1u << 8) | (1u << 9);

    RCC_CFGR = (0u << 4) | (5u << 10) | (4u << 13);  /* AHB/1 APB1/4 APB2/2 */
    RCC_CFGR |= 2u;                                   /* SYSCLK = PLL */
    for (t = 0; t < 2000000u; t++) if (((RCC_CFGR >> 2) & 3u) == 2u) break;
    return (((RCC_CFGR >> 2) & 3u) == 2u);
}

/* ---- caches ----------------------------------------------------------- */
static void icache_enable(void)
{
    DSB(); ISB();
    SCB_ICIALLU = 0u;
    DSB(); ISB();
    SCB_CCR |= CCR_IC;
    DSB(); ISB();
}

static void icache_disable(void)
{
    DSB(); ISB();
    SCB_CCR &= ~CCR_IC;
    SCB_ICIALLU = 0u;
    DSB(); ISB();
}

static void dcache_invalidate_all(void)
{
    uint32_t ccsidr, sets, ways, s, w, shift;
    SCB_CSSELR = 0u; DSB();
    ccsidr = SCB_CCSIDR;
    sets = ((ccsidr >> 13) & 0x7FFFu);
    ways = ((ccsidr >> 3) & 0x3FFu);
    shift = 32u - 2u;                       /* 4-way on this part */
    for (w = 0; w <= ways; w++)
        for (s = 0; s <= sets; s++)
            SCB_DCISW = (w << shift) | (s << 5);
    DSB();
}

static void dcache_enable(void)
{
    dcache_invalidate_all();
    SCB_CCR |= CCR_DC;
    DSB(); ISB();
}

static void dcache_disable(void)
{
    uint32_t ccsidr, sets, ways, s, w, shift;
    DSB();
    SCB_CCR &= ~CCR_DC;
    DSB();
    SCB_CSSELR = 0u; DSB();
    ccsidr = SCB_CCSIDR;
    sets = ((ccsidr >> 13) & 0x7FFFu);
    ways = ((ccsidr >> 3) & 0x3FFu);
    shift = 32u - 2u;
    for (w = 0; w <= ways; w++)
        for (s = 0; s <= sets; s++)
            SCB_DCCISW = (w << shift) | (s << 5);
    DSB(); ISB();
}

/* ---- cache-state randomisation ---------------------------------------- */
#define SCRATCH_WORDS 8192u                 /* 32 KB, well above the 4 KB D-cache */
static volatile uint32_t scratch[SCRATCH_WORDS];

static void randomise_cache_state(void)
{
    uint32_t i, acc = 0;
    uint32_t n = 32u + (tacle_rand() % 224u);   /* variable eviction depth */
    for (i = 0; i < n; i++)
        acc += scratch[tacle_rand() % SCRATCH_WORDS];
    scratch[tacle_rand() % SCRATCH_WORDS] = acc;
}

/* ---- benchmark table --------------------------------------------------- */
typedef struct {
    const char *name;
    void (*init)(void);
    void (*run)(void);
    void (*randomise)(void);
    uint32_t n_samples;     /* per-benchmark: heavy kernels get fewer */
} bench_t;

/* Randomisers run AFTER init(). The kernel initialisers rewrite their inputs
   deterministically, so randomising first would simply be overwritten -- that
   was the defect in the first campaign, which left matrix1 and binarysearch
   with two distinct execution times each.

   Numerical kernels are *perturbed* rather than replaced, so the matrices stay
   well conditioned and the kernel keeps computing something meaningful. */
static void rnd_bsort(void) { /* patched initialiser already calls tacle_rand() */ }

static void rnd_insertsort(void)
{
    int i;
    for (i = 1; i <= 10; i++)
        insertsort_a[i] = (unsigned int)(tacle_rand() & 0xFFFFu);
}

static double ludcmp_a0[50][50];
static int ludcmp_saved = 0;
static void rnd_ludcmp(void)
{
    int i, j;
    /* Capture the pristine matrix once, then perturb from it each activation.
       The previous version used += on the diagonal, which accumulated across
       activations and eventually produced a non-terminating solve at large
       sample counts. */
    if (!ludcmp_saved) {
        for (i = 0; i < 50; i++)
            for (j = 0; j < 50; j++) ludcmp_a0[i][j] = ludcmp_a[i][j];
        ludcmp_saved = 1;
    }
    for (i = 0; i < 50; i++)
        for (j = 0; j < 50; j++) ludcmp_a[i][j] = ludcmp_a0[i][j];
    for (i = 0; i < 50; i++)
        ludcmp_a[i][i] += (double)(int)((tacle_rand() & 0x3Fu)) * 0.01;
}

static void rnd_fft(void)
{
    int i;
    for (i = 0; i < 2048; i++)
        fft_input_data[i] += (int)(tacle_rand() & 0x3Fu) - 32;
}

static void rnd_quicksort(void)
{
    int t, j, a, b; char tmp;
    for (t = 0; t < 256; t++) {
        a = (int)(tacle_rand() % 681u); b = (int)(tacle_rand() % 681u);
        if (a == b) continue;
        for (j = 0; j < 20; j++) {
            tmp = quicksort_strings[a][j];
            quicksort_strings[a][j] = quicksort_strings[b][j];
            quicksort_strings[b][j] = tmp;
        }
    }
}

/* fft is excluded: with caches disabled it produced two distinct execution
   times across 3000 activations, because a fixed-size 1024-point transform
   does identical work every activation, and perturbing the input changes
   values but not the executed path. Neither CP nor EVT applies to that.

   quicksort runs ~7.1M cycles per activation with caches disabled, so it gets
   20,000 activations rather than 200,000; that still supports a
   training-conditional bound to alpha ~ 3e-4 and a marginal bound to 1e-4. */
static const bench_t BENCHES[] = {
    { "quicksort",  quicksort_init,  quicksort_main,  rnd_quicksort,  20000u },
};
#define N_BENCH (sizeof(BENCHES) / sizeof(BENCHES[0]))

/* ---- one measurement block --------------------------------------------- */
static void run_block(const bench_t *b, int caches_on)
{
    uint32_t i, t0, t1, d;

    uart_puts("#task,");
    uart_puts(b->name);
    uart_puts(caches_on ? "_cache_on" : "_cache_off");
    uart_puts("\n#n,");
    uart_putu32(b->n_samples);
    uart_puts("\n#unit,cycles\n#clock_hz,");
    uart_putu32(SYSCLK_HZ);
    uart_puts("\n#caches,");
    uart_puts(caches_on ? "icache+dcache" : "disabled");
    uart_puts("\n");
    uint32_t block_t0 = DWT_CYCCNT;

    for (i = 0; i < N_WARMUP; i++) {
        b->init(); b->randomise(); randomise_cache_state();
        b->run();
    }

    for (i = 0; i < b->n_samples; i++) {
        b->init();
        b->randomise();
        randomise_cache_state();

        DSB(); ISB();
        t0 = DWT_CYCCNT;
        __asm__ volatile("" ::: "memory");
        b->run();
        __asm__ volatile("" ::: "memory");
        t1 = DWT_CYCCNT;
        DSB();

        d = t1 - t0;                        /* correct across a 32-bit wrap */
        uart_putu32(d);
        uart_puts("\n");
        while (!(USART3_ISR & (1u << 6))) { }   /* wait for transmit complete */
    }
    uart_puts("#block_cycles,");
    uart_putu32(DWT_CYCCNT - block_t0);
    uart_puts("\n#end\n");
}

/* ---- init -------------------------------------------------------------- */
static void periph_init(void)
{
    RCC_AHB1ENR |= (1u << 1) | (1u << 3);   /* GPIOB, GPIOD */
    RCC_APB1ENR |= (1u << 18);              /* USART3 */

    GPIO_MODER(GPIOB_BASE) &= ~(3u << (0 * 2));
    GPIO_MODER(GPIOB_BASE) |=  (1u << (0 * 2));   /* LD1 output */

    GPIO_MODER(GPIOD_BASE) &= ~((3u << (8 * 2)) | (3u << (9 * 2)));
    GPIO_MODER(GPIOD_BASE) |=  ((2u << (8 * 2)) | (2u << (9 * 2)));  /* PD8 TX, PD9 RX */
    GPIO_OSPEEDR(GPIOD_BASE) |= (3u << (8 * 2));
    GPIO_AFRH(GPIOD_BASE) &= ~0xFFu;
    GPIO_AFRH(GPIOD_BASE) |=  (7u | (7u << 4));

    USART3_CR1 = 0;
    USART3_BRR = (APB1_HZ + BAUD / 2u) / BAUD;
    USART3_CR1 = (1u << 3) | (1u << 2) | (1u << 0);   /* TE | RE | UE */

    DEMCR |= (1u << 24);
    DWT_LAR = DWT_LAR_KEY;                  /* Cortex-M7 lock */
    DWT_CYCCNT = 0;
    DWT_CTRL |= (1u << 0);
}

#define SCB_SHCSR (*(volatile uint32_t *)0xE000ED24u)
#define SCB_CFSR  (*(volatile uint32_t *)0xE000ED28u)
#define SCB_HFSR  (*(volatile uint32_t *)0xE000ED2Cu)

void fault_report(const char *name)
{
    uart_puts("\n#FAULT,");
    uart_puts(name);
    uart_puts("\n#CFSR,");
    uart_putu32(SCB_CFSR);
    uart_puts("\n#HFSR,");
    uart_putu32(SCB_HFSR);
    uart_puts("\n#halted\n");
    for (;;) { }
}

int main(void)
{
    uint32_t i;
    CPACR |= (0xFu << 20);                  /* enable CP10/CP11 (FPU) */
    DSB(); ISB();
    SCB_SHCSR |= (1u << 16) | (1u << 17) | (1u << 18);  /* separate fault handlers */
    int ok = clock_to_216mhz();
    periph_init();

    /* Handshake: idle until the host opens the port and sends a byte. */
    {
        uint32_t blink = 0;
        while (!(USART3_ISR & (1u << 5))) {
            if ((++blink & 0x3FFFFu) == 0) {
                GPIO_BSRR(GPIOB_BASE) = (blink & 0x40000u) ? (1u << 0) : (1u << 16);
            }
        }
        (void)USART3_RDR;
    }

    uart_puts("\n#build,measure-12\n#clock_ok,");
    uart_puts(ok ? "216MHz" : "FAILED-fell-back");
    uart_puts("\n");

    /* Confirm the counter before spending time on a campaign */
    {
        uint32_t a = DWT_CYCCNT;
        volatile uint32_t acc = 0;
        for (i = 0; i < 1000u; i++) acc += i;
        uart_puts("#dwt,");
        uart_puts((DWT_CYCCNT != a) ? "running" : "DEAD");
        uart_puts("\n");
    }

    for (i = 0; i < SCRATCH_WORDS; i++) scratch[i] = i * 2654435761u;

    for (i = 0; i < N_BENCH; i++) {
        icache_enable(); dcache_enable();
        GPIO_BSRR(GPIOB_BASE) = (1u << 0);
        run_block(&BENCHES[i], 1);

        dcache_disable(); icache_disable();
        GPIO_BSRR(GPIOB_BASE) = (1u << 16);
        run_block(&BENCHES[i], 0);
    }

    uart_puts("#campaign_complete\n");
    for (;;) {
        GPIO_BSRR(GPIOB_BASE) = (1u << 0);
        for (i = 0; i < 4000000u; i++) __asm__ volatile("nop");
        GPIO_BSRR(GPIOB_BASE) = (1u << 16);
        for (i = 0; i < 4000000u; i++) __asm__ volatile("nop");
    }
}
