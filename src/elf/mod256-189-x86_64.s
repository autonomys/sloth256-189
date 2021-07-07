.text	

.globl	sqrx_n_mul_mod_256_189
.hidden	sqrx_n_mul_mod_256_189
.type	sqrx_n_mul_mod_256_189,@function
.align	32
sqrx_n_mul_mod_256_189:
.cfi_startproc
	.byte	0xf3,0x0f,0x1e,0xfa


	pushq	%rbp
.cfi_adjust_cfa_offset	8
.cfi_offset	%rbp,-16
	pushq	%rbx
.cfi_adjust_cfa_offset	8
.cfi_offset	%rbx,-24
	pushq	%r12
.cfi_adjust_cfa_offset	8
.cfi_offset	%r12,-32
	pushq	%r13
.cfi_adjust_cfa_offset	8
.cfi_offset	%r13,-40
	pushq	%r14
.cfi_adjust_cfa_offset	8
.cfi_offset	%r14,-48
	pushq	%r15
.cfi_adjust_cfa_offset	8
.cfi_offset	%r15,-56
	pushq	%rdi
.cfi_adjust_cfa_offset	8
.cfi_offset	%rdi,-64
	pushq	%rcx
.cfi_adjust_cfa_offset	8
.cfi_offset	%rcx,-72
	leaq	-8(%rsp),%rsp
.cfi_adjust_cfa_offset	8

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
.cfi_endproc	
.size	sqrx_n_mul_mod_256_189,.-sqrx_n_mul_mod_256_189

.globl	mulx_mod_256_189
.hidden	mulx_mod_256_189
.type	mulx_mod_256_189,@function
.align	32
mulx_mod_256_189:
.cfi_startproc
	.byte	0xf3,0x0f,0x1e,0xfa


	pushq	%rbp
.cfi_adjust_cfa_offset	8
.cfi_offset	%rbp,-16
	pushq	%rbx
.cfi_adjust_cfa_offset	8
.cfi_offset	%rbx,-24
	pushq	%r12
.cfi_adjust_cfa_offset	8
.cfi_offset	%r12,-32
	pushq	%r13
.cfi_adjust_cfa_offset	8
.cfi_offset	%r13,-40
	pushq	%r14
.cfi_adjust_cfa_offset	8
.cfi_offset	%r14,-48
	pushq	%r15
.cfi_adjust_cfa_offset	8
.cfi_offset	%r15,-56
	pushq	%rdi
.cfi_adjust_cfa_offset	8
.cfi_offset	%rdi,-64
	leaq	-16(%rsp),%rsp
.cfi_adjust_cfa_offset	16

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
.cfi_endproc	
.size	mulx_mod_256_189,.-mulx_mod_256_189

.globl	sqrx_mod_256_189
.hidden	sqrx_mod_256_189
.type	sqrx_mod_256_189,@function
.align	32
sqrx_mod_256_189:
.cfi_startproc
	.byte	0xf3,0x0f,0x1e,0xfa


	pushq	%rbp
.cfi_adjust_cfa_offset	8
.cfi_offset	%rbp,-16
	pushq	%rbx
.cfi_adjust_cfa_offset	8
.cfi_offset	%rbx,-24
	pushq	%r12
.cfi_adjust_cfa_offset	8
.cfi_offset	%r12,-32
	pushq	%r13
.cfi_adjust_cfa_offset	8
.cfi_offset	%r13,-40
	pushq	%r14
.cfi_adjust_cfa_offset	8
.cfi_offset	%r14,-48
	pushq	%r15
.cfi_adjust_cfa_offset	8
.cfi_offset	%r15,-56
	pushq	%rdi
.cfi_adjust_cfa_offset	8
.cfi_offset	%rdi,-64
	leaq	-16(%rsp),%rsp
.cfi_adjust_cfa_offset	16

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

.align	32
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
.cfi_restore	%r15
	movq	32(%rsp),%r14
.cfi_restore	%r14
	movq	40(%rsp),%r13
.cfi_restore	%r13
	movq	48(%rsp),%r12
.cfi_restore	%r12
	movq	56(%rsp),%rbx
.cfi_restore	%rbx
	movq	64(%rsp),%rbp
.cfi_restore	%rbp
	leaq	72(%rsp),%rsp
.cfi_adjust_cfa_offset	-8*9
	.byte	0xf3,0xc3
.cfi_endproc	
.size	sqrx_mod_256_189,.-sqrx_mod_256_189

.globl	redc_mod_256_189
.hidden	redc_mod_256_189
.type	redc_mod_256_189,@function
.align	32
redc_mod_256_189:
.cfi_startproc
	.byte	0xf3,0x0f,0x1e,0xfa

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

	.byte	0xf3,0xc3
.cfi_endproc
.size	redc_mod_256_189,.-redc_mod_256_189
.globl	cneg_mod_256_189
.hidden	cneg_mod_256_189
.type	cneg_mod_256_189,@function
.align	32
cneg_mod_256_189:
.cfi_startproc
	.byte	0xf3,0x0f,0x1e,0xfa

	movq	0(%rsi),%rax
	movq	8(%rsi),%r10
	movq	16(%rsi),%r9
	movq	%rax,%r11
	movq	24(%rsi),%rsi
	orq	%r10,%rax
	orq	%r9,%rax
	negq	%rdx
	orq	%rsi,%rax
	cmovnzq	%rdx,%rax

	movq	%rax,%rdx
	andq	$189,%rax

	addq	%rax,%r11
	adcq	$0,%r10
	adcq	$0,%r9
	adcq	$0,%rsi

	andq	$1,%rax
	xorq	%rdx,%r11
	xorq	%rdx,%r10
	xorq	%rdx,%r9
	xorq	%rdx,%rsi
	addq	%rax,%r11
	adcq	$0,%r10
	adcq	$0,%r9
	adcq	$0,%rsi

	movq	%r11,0(%rdi)
	movq	%r10,8(%rdi)
	movq	%r9,16(%rdi)
	movq	%rsi,24(%rdi)

	.byte	0xf3,0xc3
.cfi_endproc
.size	cneg_mod_256_189,.-cneg_mod_256_189

.globl	xor_n_check_mod_256_189
.hidden	xor_n_check_mod_256_189
.type	xor_n_check_mod_256_189,@function
.align	32
xor_n_check_mod_256_189:
.cfi_startproc
	.byte	0xf3,0x0f,0x1e,0xfa

	movq	0(%rsi),%r11
	movq	8(%rsi),%r10
	movq	16(%rsi),%r9
	movq	24(%rsi),%rsi

	xorq	0(%rdx),%r11
	xorq	8(%rdx),%r10
	xorq	16(%rdx),%r9
	xorq	24(%rdx),%rsi

	xorl	%eax,%eax
	movq	%r11,0(%rdi)
	addq	$189,%r11
	movq	%r10,8(%rdi)
	adcq	$0,%r10
	movq	%r9,16(%rdi)
	adcq	$0,%r9
	movq	%rsi,24(%rdi)
	adcq	$0,%rsi
	adcq	$0,%rax

	.byte	0xf3,0xc3
.cfi_endproc
.size	xor_n_check_mod_256_189,.-xor_n_check_mod_256_189

.globl	swap_neigh_256_189
.hidden	swap_neigh_256_189
.type	swap_neigh_256_189,@function
.align	32
swap_neigh_256_189:
.cfi_startproc
	.byte	0xf3,0x0f,0x1e,0xfa

	movq	0(%rsi),%rax
	movq	8(%rsi),%r10
	movq	16(%rsi),%r9
	movq	%rax,%r11
	andq	$1,%rax
	movq	24(%rsi),%rsi

	subq	$1,%r11
	sbbq	$0,%r10
	leaq	(%rax,%rax,1),%rax
	sbbq	$0,%r9
	sbbq	$0,%rsi

	addq	%rax,%r11
	adcq	$0,%r10
	adcq	$0,%r9
	adcq	$0,%rsi

	movq	%r11,0(%rdi)
	movq	%r10,8(%rdi)
	movq	%r9,16(%rdi)
	movq	%rsi,24(%rdi)

	.byte	0xf3,0xc3
.cfi_endproc
.size	swap_neigh_256_189,.-swap_neigh_256_189

.section	.note.gnu.property,"a",@note
	.long	4,2f-1f,5
	.byte	0x47,0x4E,0x55,0
1:	.long	0xc0000002,4,3
.align	8
2:
