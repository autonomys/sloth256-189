.text	



.globl	sqrx_n_mul_mod_256_189

.def	sqrx_n_mul_mod_256_189;	.scl 2;	.type 32;	.endef
.p2align	5
sqrx_n_mul_mod_256_189:
	.byte	0xf3,0x0f,0x1e,0xfa
	movq	%rdi,8(%rsp)
	movq	%rsi,16(%rsp)
	movq	%rsp,%r11
.LSEH_begin_sqrx_n_mul_mod_256_189:
	movq	%rcx,%rdi
	movq	%rdx,%rsi
	movq	%r8,%rdx
	movq	%r9,%rcx


	pushq	%rbp

	pushq	%rbx

	pushq	%r12

	pushq	%r13

	pushq	%r14

	pushq	%r15

	pushq	%rdi

	pushq	%rcx

	leaq	-8(%rsp),%rsp

.LSEH_body_sqrx_n_mul_mod_256_189:


	movl	%edx,%eax
	movq	0(%rsi),%rdx
	movq	8(%rsi),%rcx
	xorq	%r8,%r8
	movq	16(%rsi),%r14
	movq	24(%rsi),%r15

.Loop_sqrx:

	mulxq	%rcx,%r9,%rbx
	cmovncq	%rdx,%r8
	mulxq	%r14,%r10,%rsi
	xorq	%rdi,%rdi
	adcxq	%rbx,%r10
	mulxq	%r15,%r11,%r12
	movq	%rcx,%rdx
	adcxq	%rsi,%r11
	adcxq	%rdi,%r12


	mulxq	%r14,%rsi,%rbx
	adoxq	%rsi,%r11
	adcxq	%rbx,%r12
	mulxq	%r15,%rsi,%r13
	movq	%r14,%rdx
	adoxq	%rsi,%r12
	adcxq	%rdi,%r13


	mulxq	%r15,%rsi,%rbp
	movq	%r8,%rdx
	adoxq	%rsi,%r13
	adcxq	%rdi,%rbp
	adoxq	%rdi,%rbp


	mulxq	%rdx,%r8,%rsi
	movq	%rcx,%rdx
	adcxq	%r9,%r9
	adoxq	%rsi,%r9
	adcxq	%r10,%r10
	mulxq	%rdx,%rsi,%rbx
	movq	%r14,%rdx
	adcxq	%r11,%r11
	adoxq	%rsi,%r10
	adcxq	%r12,%r12
	adoxq	%rbx,%r11
	mulxq	%rdx,%rsi,%rbx
	movq	%r15,%rdx
	adcxq	%r13,%r13
	adoxq	%rsi,%r12
	adcxq	%rbp,%rbp
	adoxq	%rbx,%r13
	mulxq	%rdx,%r14,%r15
	movl	$189,%edx
	adoxq	%rbp,%r14
	adcxq	%rdi,%r15
	adoxq	%rdi,%r15


	mulxq	%r12,%rsi,%rbx
	adcxq	%rsi,%r8
	adoxq	%rbx,%r9
	mulxq	%r13,%rcx,%rbx
	adcxq	%r9,%rcx
	adoxq	%rbx,%r10
	mulxq	%r14,%r14,%rbx
	adcxq	%r10,%r14
	adoxq	%rbx,%r11
	mulxq	%r15,%r15,%r12
	adcxq	%r11,%r15
	adoxq	%rdi,%r12
	adcxq	%rdi,%r12

	movl	%eax,%eax
	movq	8(%rsp),%rsi
	imulq	%r12,%rdx

	addq	%r8,%rdx
	adcq	$0,%rcx
	leaq	189(%rdx),%r8
	adcq	$0,%r14
	adcq	$0,%r15

	decl	%eax
	jnz	.Loop_sqrx

	movq	%rdx,%rbp
	movq	(%rsi),%rdx
	cmovcq	%r8,%rbp

	jmp	.Lmulx_data_is_loaded
.LSEH_epilogue_sqrx_n_mul_mod_256_189:
	mov	80(%rsp),%rdi
	mov	88(%rsp),%rsi


.LSEH_end_sqrx_n_mul_mod_256_189:


.globl	mulx_mod_256_189

.def	mulx_mod_256_189;	.scl 2;	.type 32;	.endef
.p2align	5
mulx_mod_256_189:
	.byte	0xf3,0x0f,0x1e,0xfa
	movq	%rdi,8(%rsp)
	movq	%rsi,16(%rsp)
	movq	%rsp,%r11
.LSEH_begin_mulx_mod_256_189:
	movq	%rcx,%rdi
	movq	%rdx,%rsi
	movq	%r8,%rdx


	pushq	%rbp

	pushq	%rbx

	pushq	%r12

	pushq	%r13

	pushq	%r14

	pushq	%r15

	pushq	%rdi

	leaq	-16(%rsp),%rsp

.LSEH_body_mulx_mod_256_189:


	movq	%rdx,%rax
	movq	0(%rdx),%rbp
	movq	0(%rsi),%rdx
	movq	8(%rax),%rcx
	movq	16(%rax),%r14
	movq	24(%rax),%r15

.Lmulx_data_is_loaded:
	mulxq	%rbp,%r8,%rax
	xorl	%edi,%edi
	mulxq	%rcx,%r9,%rbx
	adcxq	%rax,%r9
	mulxq	%r14,%r10,%rax
	adcxq	%rbx,%r10
	mulxq	%r15,%r11,%r12
	movq	8(%rsi),%rdx
	adcxq	%rax,%r11
	movq	%r14,(%rsp)
	adcxq	%rdi,%r12

	mulxq	%rbp,%rax,%rbx
	adoxq	%rax,%r9
	adcxq	%rbx,%r10
	mulxq	%rcx,%rax,%rbx
	adoxq	%rax,%r10
	adcxq	%rbx,%r11
	mulxq	%r14,%rax,%rbx
	adoxq	%rax,%r11
	adcxq	%rbx,%r12
	mulxq	%r15,%rax,%r13
	movq	16(%rsi),%rdx
	adoxq	%rax,%r12
	adcxq	%rdi,%r13
	adoxq	%rdi,%r13

	mulxq	%rbp,%rax,%rbx
	adcxq	%rax,%r10
	adoxq	%rbx,%r11
	mulxq	%rcx,%rax,%rbx
	adcxq	%rax,%r11
	adoxq	%rbx,%r12
	mulxq	%r14,%rax,%rbx
	adcxq	%rax,%r12
	adoxq	%rbx,%r13
	mulxq	%r15,%rax,%r14
	movq	24(%rsi),%rdx
	adcxq	%rax,%r13
	adoxq	%rdi,%r14
	adcxq	%rdi,%r14

	mulxq	%rbp,%rax,%rbx
	adoxq	%rax,%r11
	adcxq	%rbx,%r12
	mulxq	%rcx,%rax,%rbx
	adoxq	%rax,%r12
	adcxq	%rbx,%r13
	mulxq	(%rsp),%rax,%rbx
	adoxq	%rax,%r13
	adcxq	%rbx,%r14
	mulxq	%r15,%rax,%r15
	movl	$189,%edx
	adoxq	%rax,%r14
	adcxq	%rdi,%r15
	adoxq	%rdi,%r15

	jmp	.Lreduce64
.LSEH_epilogue_mulx_mod_256_189:
	mov	80(%rsp),%rdi
	mov	88(%rsp),%rsi


.LSEH_end_mulx_mod_256_189:


.globl	sqrx_mod_256_189

.def	sqrx_mod_256_189;	.scl 2;	.type 32;	.endef
.p2align	5
sqrx_mod_256_189:
	.byte	0xf3,0x0f,0x1e,0xfa
	movq	%rdi,8(%rsp)
	movq	%rsi,16(%rsp)
	movq	%rsp,%r11
.LSEH_begin_sqrx_mod_256_189:
	movq	%rcx,%rdi
	movq	%rdx,%rsi


	pushq	%rbp

	pushq	%rbx

	pushq	%r12

	pushq	%r13

	pushq	%r14

	pushq	%r15

	pushq	%rdi

	leaq	-16(%rsp),%rsp

.LSEH_body_sqrx_mod_256_189:


	movq	0(%rsi),%rdx
	movq	8(%rsi),%rcx
	movq	16(%rsi),%rbp
	movq	24(%rsi),%rsi


	mulxq	%rdx,%r8,%r15
	mulxq	%rcx,%r9,%rax
	xorl	%edi,%edi
	mulxq	%rbp,%r10,%rbx
	adcxq	%rax,%r10
	mulxq	%rsi,%r11,%r12
	movq	%rcx,%rdx
	adcxq	%rbx,%r11
	adcxq	%rdi,%r12


	mulxq	%rbp,%rax,%rbx
	adoxq	%rax,%r11
	adcxq	%rbx,%r12
	mulxq	%rsi,%rax,%r13
	movq	%rbp,%rdx
	adoxq	%rax,%r12
	adcxq	%rdi,%r13


	mulxq	%rsi,%rax,%r14
	movq	%rcx,%rdx
	adoxq	%rax,%r13
	adcxq	%rdi,%r14
	adoxq	%rdi,%r14

	adcxq	%r9,%r9
	adoxq	%r15,%r9
	adcxq	%r10,%r10
	mulxq	%rdx,%rax,%rbx
	movq	%rbp,%rdx
	adcxq	%r11,%r11
	adoxq	%rax,%r10
	adcxq	%r12,%r12
	adoxq	%rbx,%r11
	mulxq	%rdx,%rax,%rbx
	movq	%rsi,%rdx
	adcxq	%r13,%r13
	adoxq	%rax,%r12
	adcxq	%r14,%r14
	adoxq	%rbx,%r13
	mulxq	%rdx,%rax,%r15
	movl	$189,%edx
	adoxq	%rax,%r14
	adcxq	%rdi,%r15
	adoxq	%rdi,%r15
	jmp	.Lreduce64

.p2align	5
.Lreduce64:
	mulxq	%r12,%rax,%rbx
	adcxq	%rax,%r8
	adoxq	%rbx,%r9
	mulxq	%r13,%rax,%rbx
	adcxq	%rax,%r9
	adoxq	%rbx,%r10
	mulxq	%r14,%rax,%rbx
	adcxq	%rax,%r10
	adoxq	%rbx,%r11
	mulxq	%r15,%rax,%r12
	adcxq	%rax,%r11
	adoxq	%rdi,%r12
	adcxq	%rdi,%r12

	movq	16(%rsp),%rdi
	imulq	%rdx,%r12

	addq	%r12,%r8
	adcq	$0,%r9
	adcq	$0,%r10
	adcq	$0,%r11

	leaq	189(%r8),%rax
	cmovcq	%rax,%r8

	movq	%r9,8(%rdi)
	movq	%r10,16(%rdi)
	movq	%r11,24(%rdi)
	movq	%r8,0(%rdi)

	movq	24(%rsp),%r15

	movq	32(%rsp),%r14

	movq	40(%rsp),%r13

	movq	48(%rsp),%r12

	movq	56(%rsp),%rbx

	movq	64(%rsp),%rbp

	leaq	72(%rsp),%rsp

.LSEH_epilogue_sqrx_mod_256_189:
	mov	8(%rsp),%rdi
	mov	16(%rsp),%rsi

	.byte	0xf3,0xc3

.LSEH_end_sqrx_mod_256_189:


.globl	redc_mod_256_189

.def	redc_mod_256_189;	.scl 2;	.type 32;	.endef
.p2align	5
redc_mod_256_189:
	.byte	0xf3,0x0f,0x1e,0xfa
	movq	%rdi,8(%rsp)
	movq	%rsi,16(%rsp)
	movq	%rsp,%r11
.LSEH_begin_redc_mod_256_189:
	movq	%rcx,%rdi
	movq	%rdx,%rsi


.LSEH_body_redc_mod_256_189:

	movq	0(%rsi),%r8
	movq	8(%rsi),%r9
	movq	16(%rsi),%r10
	movq	24(%rsi),%r11

	movq	%r8,%rax
	movq	%r9,%rdx
	movq	%r10,%rcx
	movq	%r11,%rsi

	addq	$189,%r8
	adcq	$0,%r9
	adcq	$0,%r10
	adcq	$0,%r11

	cmovncq	%rax,%r8
	cmovncq	%rdx,%r9
	cmovncq	%rcx,%r10
	cmovncq	%rsi,%r11

	movq	%r8,0(%rdi)
	movq	%r9,8(%rdi)
	movq	%r10,16(%rdi)
	movq	%r11,24(%rdi)

.LSEH_epilogue_redc_mod_256_189:
	mov	8(%rsp),%rdi
	mov	16(%rsp),%rsi

	.byte	0xf3,0xc3

.LSEH_end_redc_mod_256_189:

.globl	cneg_mod_256_189

.def	cneg_mod_256_189;	.scl 2;	.type 32;	.endef
.p2align	5
cneg_mod_256_189:
	.byte	0xf3,0x0f,0x1e,0xfa

	movq	0(%rdx),%rax
	movq	8(%rdx),%r10
	movq	16(%rdx),%r9
	movq	%rax,%r11
	movq	24(%rdx),%rdx
	orq	%r10,%rax
	orq	%r9,%rax
	negq	%r8
	orq	%rdx,%rax
	cmovnzq	%r8,%rax

	movq	%rax,%r8
	andq	$189,%rax

	addq	%rax,%r11
	adcq	$0,%r10
	adcq	$0,%r9
	adcq	$0,%rdx

	andq	$1,%rax
	xorq	%r8,%r11
	xorq	%r8,%r10
	xorq	%r8,%r9
	xorq	%r8,%rdx
	addq	%rax,%r11
	adcq	$0,%r10
	adcq	$0,%r9
	adcq	$0,%rdx

	movq	%r11,0(%rcx)
	movq	%r10,8(%rcx)
	movq	%r9,16(%rcx)
	movq	%rdx,24(%rcx)

	.byte	0xf3,0xc3



.globl	xor_n_check_mod_256_189

.def	xor_n_check_mod_256_189;	.scl 2;	.type 32;	.endef
.p2align	5
xor_n_check_mod_256_189:
	.byte	0xf3,0x0f,0x1e,0xfa

	movq	0(%rdx),%r11
	movq	8(%rdx),%r10
	movq	16(%rdx),%r9
	movq	24(%rdx),%rdx

	xorq	0(%r8),%r11
	xorq	8(%r8),%r10
	xorq	16(%r8),%r9
	xorq	24(%r8),%rdx

	xorl	%eax,%eax
	movq	%r11,0(%rcx)
	addq	$189,%r11
	movq	%r10,8(%rcx)
	adcq	$0,%r10
	movq	%r9,16(%rcx)
	adcq	$0,%r9
	movq	%rdx,24(%rcx)
	adcq	$0,%rdx
	adcq	$0,%rax

	.byte	0xf3,0xc3



.globl	swap_neigh_256_189

.def	swap_neigh_256_189;	.scl 2;	.type 32;	.endef
.p2align	5
swap_neigh_256_189:
	.byte	0xf3,0x0f,0x1e,0xfa

	movq	0(%rdx),%rax
	movq	8(%rdx),%r10
	movq	16(%rdx),%r9
	movq	%rax,%r11
	andq	$1,%rax
	movq	24(%rdx),%rdx

	subq	$1,%r11
	sbbq	$0,%r10
	leaq	(%rax,%rax,1),%rax
	sbbq	$0,%r9
	sbbq	$0,%rdx

	addq	%rax,%r11
	adcq	$0,%r10
	adcq	$0,%r9
	adcq	$0,%rdx

	movq	%r11,0(%rcx)
	movq	%r10,8(%rcx)
	movq	%r9,16(%rcx)
	movq	%rdx,24(%rcx)

	.byte	0xf3,0xc3

.section	.pdata
.p2align	2
.rva	.LSEH_begin_sqrx_n_mul_mod_256_189
.rva	.LSEH_body_sqrx_n_mul_mod_256_189
.rva	.LSEH_info_sqrx_n_mul_mod_256_189_prologue

.rva	.LSEH_body_sqrx_n_mul_mod_256_189
.rva	.LSEH_epilogue_sqrx_n_mul_mod_256_189
.rva	.LSEH_info_sqrx_n_mul_mod_256_189_body

.rva	.LSEH_epilogue_sqrx_n_mul_mod_256_189
.rva	.LSEH_end_sqrx_n_mul_mod_256_189
.rva	.LSEH_info_sqrx_n_mul_mod_256_189_epilogue

.rva	.LSEH_begin_mulx_mod_256_189
.rva	.LSEH_body_mulx_mod_256_189
.rva	.LSEH_info_mulx_mod_256_189_prologue

.rva	.LSEH_body_mulx_mod_256_189
.rva	.LSEH_epilogue_mulx_mod_256_189
.rva	.LSEH_info_mulx_mod_256_189_body

.rva	.LSEH_epilogue_mulx_mod_256_189
.rva	.LSEH_end_mulx_mod_256_189
.rva	.LSEH_info_mulx_mod_256_189_epilogue

.rva	.LSEH_begin_sqrx_mod_256_189
.rva	.LSEH_body_sqrx_mod_256_189
.rva	.LSEH_info_sqrx_mod_256_189_prologue

.rva	.LSEH_body_sqrx_mod_256_189
.rva	.LSEH_epilogue_sqrx_mod_256_189
.rva	.LSEH_info_sqrx_mod_256_189_body

.rva	.LSEH_epilogue_sqrx_mod_256_189
.rva	.LSEH_end_sqrx_mod_256_189
.rva	.LSEH_info_sqrx_mod_256_189_epilogue

.rva	.LSEH_begin_redc_mod_256_189
.rva	.LSEH_body_redc_mod_256_189
.rva	.LSEH_info_redc_mod_256_189_prologue

.rva	.LSEH_body_redc_mod_256_189
.rva	.LSEH_epilogue_redc_mod_256_189
.rva	.LSEH_info_redc_mod_256_189_body

.rva	.LSEH_epilogue_redc_mod_256_189
.rva	.LSEH_end_redc_mod_256_189
.rva	.LSEH_info_redc_mod_256_189_epilogue

.section	.xdata
.p2align	3
.LSEH_info_sqrx_n_mul_mod_256_189_prologue:
.byte	1,0,5,0x0b
.byte	0,0x74,1,0
.byte	0,0x64,2,0
.byte	0,0x03
.byte	0,0
.LSEH_info_sqrx_n_mul_mod_256_189_body:
.byte	1,0,19,0
.byte	0x00,0x14,0x01,0x00
.byte	0x00,0xf4,0x03,0x00
.byte	0x00,0xe4,0x04,0x00
.byte	0x00,0xd4,0x05,0x00
.byte	0x00,0xc4,0x06,0x00
.byte	0x00,0x34,0x07,0x00
.byte	0x00,0x54,0x08,0x00
.byte	0x00,0x74,0x0a,0x00
.byte	0x00,0x64,0x0b,0x00
.byte	0x00,0x82
.byte	0x00,0x00,0x00,0x00,0x00,0x00
.LSEH_info_sqrx_n_mul_mod_256_189_epilogue:
.byte	1,0,5,0
.byte	0x00,0x74,0x0a,0x00
.byte	0x00,0x64,0x0b,0x00
.byte	0x00,0x82
.byte	0x00,0x00

.LSEH_info_mulx_mod_256_189_prologue:
.byte	1,0,5,0x0b
.byte	0,0x74,1,0
.byte	0,0x64,2,0
.byte	0,0x03
.byte	0,0
.LSEH_info_mulx_mod_256_189_body:
.byte	1,0,17,0
.byte	0x00,0xf4,0x03,0x00
.byte	0x00,0xe4,0x04,0x00
.byte	0x00,0xd4,0x05,0x00
.byte	0x00,0xc4,0x06,0x00
.byte	0x00,0x34,0x07,0x00
.byte	0x00,0x54,0x08,0x00
.byte	0x00,0x74,0x0a,0x00
.byte	0x00,0x64,0x0b,0x00
.byte	0x00,0x82
.byte	0x00,0x00
.LSEH_info_mulx_mod_256_189_epilogue:
.byte	1,0,5,0
.byte	0x00,0x74,0x0a,0x00
.byte	0x00,0x64,0x0b,0x00
.byte	0x00,0x82
.byte	0x00,0x00

.LSEH_info_sqrx_mod_256_189_prologue:
.byte	1,0,5,0x0b
.byte	0,0x74,1,0
.byte	0,0x64,2,0
.byte	0,0x03
.byte	0,0
.LSEH_info_sqrx_mod_256_189_body:
.byte	1,0,17,0
.byte	0x00,0xf4,0x03,0x00
.byte	0x00,0xe4,0x04,0x00
.byte	0x00,0xd4,0x05,0x00
.byte	0x00,0xc4,0x06,0x00
.byte	0x00,0x34,0x07,0x00
.byte	0x00,0x54,0x08,0x00
.byte	0x00,0x74,0x0a,0x00
.byte	0x00,0x64,0x0b,0x00
.byte	0x00,0x82
.byte	0x00,0x00
.LSEH_info_sqrx_mod_256_189_epilogue:
.byte	1,0,4,0
.byte	0x00,0x74,0x01,0x00
.byte	0x00,0x64,0x02,0x00
.byte	0x00,0x00,0x00,0x00

.LSEH_info_redc_mod_256_189_prologue:
.byte	1,0,5,0x0b
.byte	0,0x74,1,0
.byte	0,0x64,2,0
.byte	0,0x03
.byte	0,0
.LSEH_info_redc_mod_256_189_body:
.byte	1,0,4,0
.byte	0x00,0x74,0x01,0x00
.byte	0x00,0x64,0x02,0x00
.byte	0x00,0x00,0x00,0x00
.LSEH_info_redc_mod_256_189_epilogue:
.byte	1,0,4,0
.byte	0x00,0x74,0x01,0x00
.byte	0x00,0x64,0x02,0x00
.byte	0x00,0x00,0x00,0x00

