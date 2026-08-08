#include <stdint.h>
extern uint32_t _estack, _etext, _sdata, _edata, _sbss, _ebss;
int main(void);
void fault_report(const char *name);      /* provided by meas_main8.c */

void Reset_Handler(void){
    uint32_t *src=&_etext,*dst=&_sdata;
    while(dst<&_edata) *dst++=*src++;
    for(dst=&_sbss; dst<&_ebss;) *dst++=0;
    main();
    for(;;){}
}
/* Named fault handlers. A silent infinite loop here is indistinguishable from
   a slow benchmark, which is exactly how the fpv5-d16 misconfiguration went
   undiagnosed across three campaign runs. */
void NMI_Handler(void)        { fault_report("NMI"); }
void HardFault_Handler(void)  { fault_report("HardFault"); }
void MemManage_Handler(void)  { fault_report("MemManage"); }
void BusFault_Handler(void)   { fault_report("BusFault"); }
void UsageFault_Handler(void) { fault_report("UsageFault"); }
void Default_Handler(void)    { fault_report("Unexpected"); }

__attribute__((section(".isr_vector"),used))
void (* const g_vectors[])(void) = {
    (void(*)(void))&_estack, Reset_Handler,
    NMI_Handler, HardFault_Handler, MemManage_Handler, BusFault_Handler,
    UsageFault_Handler, 0,0,0, Default_Handler, Default_Handler, 0,
    Default_Handler, Default_Handler
};
