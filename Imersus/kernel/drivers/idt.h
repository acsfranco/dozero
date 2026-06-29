#pragma once
#include <stdint.h>

extern void isr0();
extern void isr1();
extern void isr2();
extern void isr3();
extern void isr4();
extern void isr5();
extern void isr6();
extern void isr7();
extern void isr8();
extern void isr9();
extern void isr10();
extern void isr11();
extern void isr12();
extern void isr13();
extern void isr14();
extern void isr15();
extern void isr16();
extern void isr17();
extern void isr18();
extern void isr19();
extern void isr20();
extern void isr21();
extern void isr22();
extern void isr23();
extern void isr24();
extern void isr25();
extern void isr26();
extern void isr27();
extern void isr28();
extern void isr29();
extern void isr30();
extern void isr31();
extern void irq32();
extern void irq33();

typedef struct {
  uint16_t offset_lowerbits; // Endereço de offset menos significativo do interrupt handle
  uint16_t selector;         // Seletor do segmento de código definido na GDT - 0x8
  uint8_t  zero;             // Área reservada
  uint8_t  type_attr;        // Tipo de gate, DPL, Present Bit
  uint16_t offset_higherbits;// Endereço de offset mais significativo do interrup handle
} __attribute__((packed)) idt_entry_t;

typedef struct {
  uint16_t limit;
  uint32_t base;
} __attribute__((packed)) idtr_t;

void idt_set_gate(uint8_t, uint32_t, uint16_t, uint8_t);
void load_idt();
void idt_init();
