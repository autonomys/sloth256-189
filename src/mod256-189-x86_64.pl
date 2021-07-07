#!/usr/bin/env perl

$flavour = shift;
$output  = shift;
if ($flavour =~ /\./) { $output = $flavour; undef $flavour; }

$win64=0; $win64=1 if ($flavour =~ /[nm]asm|mingw64/ || $output =~ /\.asm$/);

$0 =~ m/(.*[\/\\])[^\/\\]+$/; $dir=$1;
( $xlate="${dir}x86_64-xlate.pl" and -f $xlate ) or
( $xlate="${dir}../../perlasm/x86_64-xlate.pl" and -f $xlate) or
die "can't locate x86_64-xlate.pl";

open STDOUT,"| \"$^X\" \"$xlate\" $flavour \"$output\"";

$0 =~ /256\-([0-9]+)/; $n=$1; die if $n >= 1<<31;

my @acc = map("%r$_",(8..15));

$code.=<<___;
.text

.globl	sqrx_n_mul_mod_256_$n
.hidden	sqrx_n_mul_mod_256_$n
.type	sqrx_n_mul_mod_256_$n,\@function,4
.align	32
sqrx_n_mul_mod_256_$n:
.cfi_startproc
	push	%rbp
.cfi_push	%rbp
	push	%rbx
.cfi_push	%rbx
	push	%r12
.cfi_push	%r12
	push	%r13
.cfi_push	%r13
	push	%r14
.cfi_push	%r14
	push	%r15
.cfi_push	%r15
	push	%rdi			# offload dst
.cfi_push	%rdi
	push	%rcx			# offload b_ptr
.cfi_push	%rcx
	lea	-8(%rsp),%rsp
.cfi_adjust_cfa_offset	8

	mov	%edx,%eax		# counter
	mov	8*0(%rsi),%rdx		# a[0]
	mov	8*1(%rsi),%rcx		# a[1]
	xor	@acc[0],@acc[0]		# cf=0
	mov	8*2(%rsi),@acc[6]	# a[2]
	mov	8*3(%rsi),@acc[7]	# a[3]

.Loop_sqrx:
	################################################################
	mulx	%rcx,@acc[1],%rbx	# a[0]*a[1]
	cmovnc	%rdx,@acc[0]		# cf ? a[0]+$n : a[0]
	mulx	@acc[6],@acc[2],%rsi	# a[0]*a[2]
	xor	%rdi,%rdi		# cf=0,of=0
	adcx	%rbx,@acc[2]
	mulx	@acc[7],@acc[3],@acc[4]	# a[0]*a[3]
	 mov	%rcx,%rdx		# a[1]
	adcx	%rsi,@acc[3]
	adcx	%rdi,@acc[4]		# cf=0

	################################################################
	mulx	@acc[6],%rsi,%rbx	# a[1]*a[2]
	adox	%rsi,@acc[3]
	adcx	%rbx,@acc[4]
	mulx	@acc[7],%rsi,@acc[5]	# a[1]*a[3]
	 mov	@acc[6],%rdx		# a[2]
	adox	%rsi,@acc[4]
	adcx	%rdi,@acc[5]

	################################################################
	mulx	@acc[7],%rsi,%rbp	# a[2]*a[3]
	 mov	@acc[0],%rdx		# a[0]
	adox	%rsi,@acc[5]
	adcx	%rdi,%rbp		# cf=0
	adox	%rdi,%rbp		# of=0

	################################################################
	mulx	%rdx,@acc[0],%rsi	# a[0]*a[0]
	 mov	%rcx,%rdx		# a[1]
	 adcx	@acc[1],@acc[1]		# acc1:6<<1
	adox	%rsi,@acc[1]
	 adcx	@acc[2],@acc[2]
	mulx	%rdx,%rsi,%rbx		# a[1]*a[1]
	 mov	@acc[6],%rdx		# a[2]
	 adcx	@acc[3],@acc[3]
	adox	%rsi,@acc[2]
	 adcx	@acc[4],@acc[4]
	adox	%rbx,@acc[3]
	mulx	%rdx,%rsi,%rbx		# a[2]*a[2]
	 mov	@acc[7],%rdx		# a[3]
	 adcx	@acc[5],@acc[5]
	adox	%rsi,@acc[4]
	 adcx	%rbp,%rbp
	adox	%rbx,@acc[5]
	mulx	%rdx,@acc[6],@acc[7]	# a[3]*a[3]
	 mov	\$$n,%edx
	adox	%rbp,@acc[6]
	adcx	%rdi,@acc[7]		# cf=0
	adox	%rdi,@acc[7]		# of=0

	################################################################
	mulx	@acc[4],%rsi,%rbx	# reduce [and harmonize register layout]
	adcx	%rsi,@acc[0]
	adox	%rbx,@acc[1]
	mulx	@acc[5],%rcx,%rbx
	adcx	@acc[1],%rcx		# @acc[1]->%rcx
	adox	%rbx,@acc[2]
	mulx	@acc[6],@acc[6],%rbx
	adcx	@acc[2],@acc[6]		# @acc[2]->@acc[6]
	adox	%rbx,@acc[3]
	mulx	@acc[7],@acc[7],@acc[4]
	adcx	@acc[3],@acc[7]		# @acc[3]->@acc[7]
	adox	%rdi,@acc[4]
	adcx	%rdi,@acc[4]

	mov	%eax,%eax		# this gives +4% on Coffee Lake?
	mov	8(%rsp),%rsi		# load b_ptr speculatively
	imulq	@acc[4],%rdx

	add	@acc[0],%rdx		# a[0]
	adc	\$0,%rcx		# a[1]
	lea	$n(%rdx),@acc[0]	# a[0]+$n
	adc	\$0,@acc[6]		# a[2]
	adc	\$0,@acc[7]		# a[3]

	dec	%eax			# preserve cf
	jnz	.Loop_sqrx

	mov	%rdx,%rbp		# harmonize register layout
	mov	(%rsi),%rdx		# a[0]
	cmovc	@acc[0],%rbp		# cf ? a[0]+$n : a[0]

	jmp	.Lmulx_data_is_loaded
.cfi_endproc
.size	sqrx_n_mul_mod_256_$n,.-sqrx_n_mul_mod_256_$n

.globl	mulx_mod_256_$n
.hidden	mulx_mod_256_$n
.type	mulx_mod_256_$n,\@function,3
.align	32
mulx_mod_256_$n:
.cfi_startproc
	push	%rbp
.cfi_push	%rbp
	push	%rbx
.cfi_push	%rbx
	push	%r12
.cfi_push	%r12
	push	%r13
.cfi_push	%r13
	push	%r14
.cfi_push	%r14
	push	%r15
.cfi_push	%r15
	push	%rdi			# offload dst
.cfi_push	%rdi
	lea	-8*2(%rsp),%rsp
.cfi_adjust_cfa_offset	16

	mov	%rdx,%rax
	mov	8*0(%rdx),%rbp		# b[0]
	mov	8*0(%rsi),%rdx		# a[0]
	mov	8*1(%rax),%rcx		# b[1]
	mov	8*2(%rax),@acc[6]	# b[2]
	mov	8*3(%rax),@acc[7]	# b[3]

.Lmulx_data_is_loaded:
	mulx	%rbp,@acc[0],%rax	# a[0]*b[0]
	xor	%edi,%edi		# cf=0,of=0
	mulx	%rcx,@acc[1],%rbx	# a[0]*b[1]
	adcx	%rax,@acc[1]
	mulx	@acc[6],@acc[2],%rax	# a[0]*b[2]
	adcx	%rbx,@acc[2]
	mulx	@acc[7],@acc[3],@acc[4]	# a[0]*b[3]
	 mov	8*1(%rsi),%rdx		# a[1]
	adcx	%rax,@acc[3]
	mov	@acc[6],(%rsp)		# offload b[2]
	adcx	%rdi,@acc[4]		# cf=0

	mulx	%rbp,%rax,%rbx		# a[1]*b[0]
	adox	%rax,@acc[1]
	adcx	%rbx,@acc[2]
	mulx	%rcx,%rax,%rbx		# a[1]*b[1]
	adox	%rax,@acc[2]
	adcx	%rbx,@acc[3]
	mulx	@acc[6],%rax,%rbx	# a[1]*b[2]
	adox	%rax,@acc[3]
	adcx	%rbx,@acc[4]
	mulx	@acc[7],%rax,@acc[5]	# a[1]*b[3]
	 mov	8*2(%rsi),%rdx		# a[2]
	adox	%rax,@acc[4]
	adcx	%rdi,@acc[5]		# cf=0
	adox	%rdi,@acc[5]		# of=0

	mulx	%rbp,%rax,%rbx		# a[2]*b[0]
	adcx	%rax,@acc[2]
	adox	%rbx,@acc[3]
	mulx	%rcx,%rax,%rbx		# a[2]*b[1]
	adcx	%rax,@acc[3]
	adox	%rbx,@acc[4]
	mulx	@acc[6],%rax,%rbx	# a[2]*b[2]
	adcx	%rax,@acc[4]
	adox	%rbx,@acc[5]
	mulx	@acc[7],%rax,@acc[6]	# a[2]*b[3]
	 mov	8*3(%rsi),%rdx		# a[3]
	adcx	%rax,@acc[5]
	adox	%rdi,@acc[6]		# of=0
	adcx	%rdi,@acc[6]		# cf=0

	mulx	%rbp,%rax,%rbx		# a[3]*b[0]
	adox	%rax,@acc[3]
	adcx	%rbx,@acc[4]
	mulx	%rcx,%rax,%rbx		# a[3]*b[1]
	adox	%rax,@acc[4]
	adcx	%rbx,@acc[5]
	mulx	(%rsp),%rax,%rbx	# a[3]*b[2]
	adox	%rax,@acc[5]
	adcx	%rbx,@acc[6]
	mulx	@acc[7],%rax,@acc[7]	# a[3]*b[3]
	 mov	\$$n,%edx
	adox	%rax,@acc[6]
	adcx	%rdi,@acc[7]		# cf=0
	adox	%rdi,@acc[7]		# of=0

	jmp	.Lreduce64
.cfi_endproc
.size	mulx_mod_256_$n,.-mulx_mod_256_$n

.globl	sqrx_mod_256_$n
.hidden	sqrx_mod_256_$n
.type	sqrx_mod_256_$n,\@function,2
.align	32
sqrx_mod_256_$n:
.cfi_startproc
	push	%rbp
.cfi_push	%rbp
	push	%rbx
.cfi_push	%rbx
	push	%r12
.cfi_push	%r12
	push	%r13
.cfi_push	%r13
	push	%r14
.cfi_push	%r14
	push	%r15
.cfi_push	%r15
	push	%rdi			# offload dst
.cfi_push	%rdi
	lea	-8*2(%rsp),%rsp
.cfi_adjust_cfa_offset	16

	mov	8*0(%rsi),%rdx		# a[0]
	mov	8*1(%rsi),%rcx		# a[1]
	mov	8*2(%rsi),%rbp		# a[2]
	mov	8*3(%rsi),%rsi		# a[3]

	################################################################
	mulx	%rdx,@acc[0],@acc[7]	# a[0]*a[0]
	mulx	%rcx,@acc[1],%rax	# a[0]*a[1]
	xor	%edi,%edi		# cf=0,of=0
	mulx	%rbp,@acc[2],%rbx	# a[0]*a[2]
	adcx	%rax,@acc[2]
	mulx	%rsi,@acc[3],@acc[4]	# a[0]*a[3]
	 mov	%rcx,%rdx		# a[1]
	adcx	%rbx,@acc[3]
	adcx	%rdi,@acc[4]		# cf=0

	################################################################
	mulx	%rbp,%rax,%rbx		# a[1]*a[2]
	adox	%rax,@acc[3]
	adcx	%rbx,@acc[4]
	mulx	%rsi,%rax,@acc[5]	# a[1]*a[3]
	 mov	%rbp,%rdx		# a[2]
	adox	%rax,@acc[4]
	adcx	%rdi,@acc[5]

	################################################################
	mulx	%rsi,%rax,@acc[6]	# a[2]*a[3]
	 mov	%rcx,%rdx		# a[1]
	adox	%rax,@acc[5]
	adcx	%rdi,@acc[6]		# cf=0
	adox	%rdi,@acc[6]		# of=0

	 adcx	@acc[1],@acc[1]		# acc1:6<<1
	adox	@acc[7],@acc[1]
	 adcx	@acc[2],@acc[2]
	mulx	%rdx,%rax,%rbx		# a[1]*a[1]
	 mov	%rbp,%rdx		# a[2]
	 adcx	@acc[3],@acc[3]
	adox	%rax,@acc[2]
	 adcx	@acc[4],@acc[4]
	adox	%rbx,@acc[3]
	mulx	%rdx,%rax,%rbx		# a[2]*a[2]
	 mov	%rsi,%rdx		# a[3]
	 adcx	@acc[5],@acc[5]
	adox	%rax,@acc[4]
	 adcx	@acc[6],@acc[6]
	adox	%rbx,@acc[5]
	mulx	%rdx,%rax,@acc[7]	# a[3]*a[3]
	 mov	\$$n,%edx
	adox	%rax,@acc[6]
	adcx	%rdi,@acc[7]		# cf=0
	adox	%rdi,@acc[7]		# of=0
	jmp	.Lreduce64

.align	32
.Lreduce64:
	mulx	@acc[4],%rax,%rbx
	adcx	%rax,@acc[0]
	adox	%rbx,@acc[1]
	mulx	@acc[5],%rax,%rbx
	adcx	%rax,@acc[1]
	adox	%rbx,@acc[2]
	mulx	@acc[6],%rax,%rbx
	adcx	%rax,@acc[2]
	adox	%rbx,@acc[3]
	mulx	@acc[7],%rax,@acc[4]
	adcx	%rax,@acc[3]
	adox	%rdi,@acc[4]
	adcx	%rdi,@acc[4]

	mov	8*2(%rsp),%rdi		# restore dst
	imulq	%rdx,@acc[4]

	add	@acc[4],@acc[0]
	adc	\$0,@acc[1]
	adc	\$0,@acc[2]
	adc	\$0,@acc[3]

	lea	$n(@acc[0]), %rax
	cmovc	%rax,@acc[0]		# %cf ? @acc[0]+$n : @acc[0]

	mov	@acc[1],8*1(%rdi)
	mov	@acc[2],8*2(%rdi)
	mov	@acc[3],8*3(%rdi)
	mov	@acc[0],8*0(%rdi)

	mov	8*3(%rsp),%r15
.cfi_restore	%r15
	mov	8*4(%rsp),%r14
.cfi_restore	%r14
	mov	8*5(%rsp),%r13
.cfi_restore	%r13
	mov	8*6(%rsp),%r12
.cfi_restore	%r12
	mov	8*7(%rsp),%rbx
.cfi_restore	%rbx
	mov	8*8(%rsp),%rbp
.cfi_restore	%rbp
	lea	8*9(%rsp),%rsp
.cfi_adjust_cfa_offset	-8*9
	ret
.cfi_endproc
.size	sqrx_mod_256_$n,.-sqrx_mod_256_$n

.globl	redc_mod_256_$n
.hidden	redc_mod_256_$n
.type	redc_mod_256_$n,\@function,2
.align	32
redc_mod_256_$n:
	mov	8*0(%rsi),@acc[0]
	mov	8*1(%rsi),@acc[1]
	mov	8*2(%rsi),@acc[2]
	mov	8*3(%rsi),@acc[3]

	mov	@acc[0],%rax
	mov	@acc[1],%rdx
	mov	@acc[2],%rcx
	mov	@acc[3],%rsi

	add	\$$n,@acc[0]
	adc	\$0,@acc[1]
	adc	\$0,@acc[2]
	adc	\$0,@acc[3]

	cmovnc	%rax,@acc[0]
	cmovnc	%rdx,@acc[1]
	cmovnc	%rcx,@acc[2]
	cmovnc	%rsi,@acc[3]

	mov	@acc[0],8*0(%rdi)
	mov	@acc[1],8*1(%rdi)
	mov	@acc[2],8*2(%rdi)
	mov	@acc[3],8*3(%rdi)

	ret
.size	redc_mod_256_$n,.-redc_mod_256_$n
___

if (0) {
$code.=<<___;
.globl	add_mod_256_$n
.hidden	add_mod_256_$n
.type	add_mod_256_$n,\@function,3
.align	32
add_mod_256_$n:
	mov	8*0(%rsi),@acc[0]
	mov	8*1(%rsi),@acc[1]
	mov	8*2(%rsi),@acc[2]
	mov	8*3(%rsi),@acc[3]
	xor	%eax,%eax
	mov	\$$n,%esi

	add	8*0(%rdx),@acc[0]
	adc	8*1(%rdx),@acc[1]
	adc	8*2(%rdx),@acc[2]
	adc	8*3(%rdx),@acc[3]

	cmovc	%esi,%eax		# %cf ? $n : 0

	add	%rax,@acc[0]
	adc	\$0,@acc[1]
	lea	$n(@acc[0]),%rax
	adc	\$0,@acc[2]
	mov	@acc[1],8*1(%rdi)
	adc	\$0,@acc[3]
	mov	@acc[2],8*2(%rdi)
	cmovc	%rax,@acc[0]		# %cf ? @acc[0]+$n : @acc[0]
	mov	@acc[3],8*3(%rdi)
	mov	@acc[0],8*0(%rdi)

	ret
.size	add_mod_256_$n,.-add_mod_256_$n

.globl	sub_mod_256_$n
.hidden	sub_mod_256_$n
.type	sub_mod_256_$n,\@function,3
.align	32
sub_mod_256_$n:
	mov	8*0(%rsi),@acc[0]
	mov	8*1(%rsi),@acc[1]
	mov	8*2(%rsi),@acc[2]
	mov	8*3(%rsi),@acc[3]
	xor	%eax,%eax
	mov	\$$n,%esi

	sub	8*0(%rdx),@acc[0]
	sbb	8*1(%rdx),@acc[1]
	sbb	8*2(%rdx),@acc[2]
	sbb	8*3(%rdx),@acc[3]

	cmovc	%esi,%eax		# %cf ? $n : 0

	sub	%rax,@acc[0]
	sbb	\$0,@acc[1]
	lea	-$n(@acc[0]),%rax
	sbb	\$0,@acc[2]
	mov	@acc[1],8*1(%rdi)
	sbb	\$0,@acc[3]
	mov	@acc[2],8*2(%rdi)
	cmovc	%rax,@acc[0]		# %cf ? @acc[0]-$n : @acc[0]
	mov	@acc[3],8*3(%rdi)
	mov	@acc[0],8*0(%rdi)

	ret
.size	sub_mod_256_$n,.-sub_mod_256_$n

.globl	xor_mod_256_$n
.globl	xor_mod_256_$n
.type	xor_mod_256_$n,\@function,3
.align	32
xor_mod_256_$n:
.cfi_startproc
	push	%r12
.cfi_push	%r12
	push	%r13
.cfi_push	%r13
	push	%r14
.cfi_push	%r14
	push	%r15
.cfi_push	%r15
	lea	-8(%rsp),%rsp
.cfi_adjust_cfa_offset	8

	mov	8*0(%rsi), @acc[0]	# load |inp|
	mov	8*1(%rsi), @acc[1]
	mov	8*2(%rsi), @acc[2]
	mov	8*3(%rsi), @acc[3]

	mov	8*0(%rdx), @acc[4]	# load |xor|
	mov	8*1(%rdx), %rcx
	mov	8*2(%rdx), %rsi
	mov	8*3(%rdx), %rdx

	xor	@acc[0], @acc[4]	# |inp|^|xor|
	xor	@acc[1], %rcx
	xor	@acc[2], %rsi
	xor	@acc[3], %rdx

	mov	@acc[4], %rax
	mov	%rcx, @acc[5]
	mov	%rsi, @acc[6]
	mov	%rdx, @acc[7]

	add	\$$n, @acc[4]		# compare the result to the modulus
	adc	\$0, %rcx
	adc	\$0, %rsi
	adc	\$0, %rdx
	sbb	%rcx, %rcx		# -1 means |inp|^|xor| >= |mod|

	mov	%rax, @acc[4]
	or	@acc[5], %rax
	not	%rcx			# 0 means |inp|^|xor| >= |mod|
	or	@acc[6], %rax
	mov	\$-1, %rdx
	or	@acc[7], %rax
	cmovnz	%rdx, %rax		# 0 means |inp|^|xor| == 0

	and	%rcx, %rax

	cmovz	@acc[0], @acc[4]	# conditionally restore original input
	cmovz	@acc[1], @acc[5]
	cmovz	@acc[2], @acc[6]
	cmovz	@acc[3], @acc[7]

	mov	@acc[4], 8*0(%rdi)
	mov	@acc[5], 8*1(%rdi)
	mov	@acc[6], 8*2(%rdi)
	mov	@acc[7], 8*3(%rdi)

	mov	8*1(%rsp),%r15
.cfi_restore	%r15
	mov	8*2(%rsp),%r14
.cfi_restore	%r14
	mov	8*3(%rsp),%r13
.cfi_restore	%r13
	mov	8*4(%rsp),%r12
.cfi_restore	%r12
	lea	8*5(%rsp),%rsp
.cfi_adjust_cfa_offset	-8*5
	ret
.cfi_endproc
.size	xor_mod_256_$n,.-xor_mod_256_$n
___
}

{
my ($out, $a_ptr, $b_ptr) = $win64 ? ("%rcx", "%rdx", "%r8")
                                   : ("%rdi", "%rsi", "%rdx");
my @acc = (map("%r$_", (11,10,9)), $a_ptr);

$code.=<<___;
.globl	cneg_mod_256_$n
.hidden	cneg_mod_256_$n
.type	cneg_mod_256_$n,\@abi-omnipotent
.align	32
cneg_mod_256_$n:
	mov	8*0($a_ptr), %rax	# load |inp|
	mov	8*1($a_ptr), @acc[1]
	mov	8*2($a_ptr), @acc[2]
	mov	%rax, @acc[0]
	mov	8*3($a_ptr), @acc[3]
	or	@acc[1], %rax		# see if |inp| is zero
	or	@acc[2], %rax
	neg	$b_ptr			# condition bit to mask
	or	@acc[3], %rax
	cmovnz	$b_ptr, %rax		# -1 means |inp| != 0 and $b_ptr != 0

	mov	%rax, $b_ptr
	and	\$$n, %rax

	add	%rax, @acc[0]		# conditionally subtract the modulus
	adc	\$0, @acc[1]
	adc	\$0, @acc[2]
	adc	\$0, @acc[3]

	and	\$1, %rax		# %rax is known to be odd if mask is set
	xor	$b_ptr, @acc[0]		# conditionally negate the result
	xor	$b_ptr, @acc[1]
	xor	$b_ptr, @acc[2]
	xor	$b_ptr, @acc[3]
	add	%rax, @acc[0]
	adc	\$0, @acc[1]
	adc	\$0, @acc[2]
	adc	\$0, @acc[3]

	mov	@acc[0], 8*0($out)
	mov	@acc[1], 8*1($out)
	mov	@acc[2], 8*2($out)
	mov	@acc[3], 8*3($out)

	ret
.size	cneg_mod_256_$n,.-cneg_mod_256_$n

.globl	xor_n_check_mod_256_$n
.hidden	xor_n_check_mod_256_$n
.type	xor_n_check_mod_256_$n,\@abi-omnipotent
.align	32
xor_n_check_mod_256_$n:
	mov	8*0($a_ptr), @acc[0]	# load |inp|
	mov	8*1($a_ptr), @acc[1]
	mov	8*2($a_ptr), @acc[2]
	mov	8*3($a_ptr), @acc[3]

	xor	8*0($b_ptr), @acc[0]	# xor |xor|
	xor	8*1($b_ptr), @acc[1]
	xor	8*2($b_ptr), @acc[2]
	xor	8*3($b_ptr), @acc[3]

	xor	%eax, %eax
	mov	@acc[0], 8*0($out)
	add	\$$n, $acc[0]
	mov	@acc[1], 8*1($out)
	adc	\$0, $acc[1]
	mov	@acc[2], 8*2($out)
	adc	\$0, $acc[2]
	mov	@acc[3], 8*3($out)
	adc	\$0, $acc[3]
	adc	\$0, %rax

	ret
.size	xor_n_check_mod_256_$n,.-xor_n_check_mod_256_$n

.globl	swap_neigh_256_$n
.hidden	swap_neigh_256_$n
.type	swap_neigh_256_$n,\@abi-omnipotent
.align	32
swap_neigh_256_$n:
	mov	8*0($a_ptr), %rax	# load |inp|
	mov	8*1($a_ptr), @acc[1]
	mov	8*2($a_ptr), @acc[2]
	mov	%rax, @acc[0]
	and	\$1, %rax		# parity bit
	mov	8*3($a_ptr), @acc[3]

	sub	\$1, @acc[0]
	sbb	\$0, @acc[1]
	lea	(%rax,%rax), %rax	# double the parity bit
	sbb	\$0, @acc[2]
	sbb	\$0, @acc[3]

	add	%rax, @acc[0]
	adc	\$0, @acc[1]
	adc	\$0, @acc[2]
	adc	\$0, @acc[3]

	mov	@acc[0], 8*0($out)
	mov	@acc[1], 8*1($out)
	mov	@acc[2], 8*2($out)
	mov	@acc[3], 8*3($out)

	ret
.size	swap_neigh_256_$n,.-swap_neigh_256_$n
___
}

#$code =~ s/\`([^\`]*)\`/eval $1/gem;
print $code;
close STDOUT;
