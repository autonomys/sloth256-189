OPTION	DOTNAME
.text$	SEGMENT ALIGN(256) 'CODE'

PUBLIC	sqrx_n_mul_mod_256_189


ALIGN	32
sqrx_n_mul_mod_256_189	PROC PUBLIC
	DB	243,15,30,250
	mov	QWORD PTR[8+rsp],rdi	;WIN64 prologue
	mov	QWORD PTR[16+rsp],rsi
	mov	rax,rsp
$L$SEH_begin_sqrx_n_mul_mod_256_189::
	mov	rdi,rcx
	mov	rsi,rdx
	mov	rdx,r8
	mov	rcx,r9



	push	rbp

	push	rbx

	push	r12

	push	r13

	push	r14

	push	r15

	push	rdi

	push	rcx

	lea	rsp,QWORD PTR[((-8))+rsp]


	mov	eax,edx
	mov	rdx,QWORD PTR[rsi]
	mov	rcx,QWORD PTR[8+rsi]
	xor	r8,r8
	mov	r14,QWORD PTR[16+rsi]
	mov	r15,QWORD PTR[24+rsi]

$L$oop_sqrx::

	mulx	rbx,r9,rcx
	cmovnc	r8,rdx
	mulx	rsi,r10,r14
	xor	rdi,rdi
	adcx	r10,rbx
	mulx	r12,r11,r15
	mov	rdx,rcx
	adcx	r11,rsi
	adcx	r12,rdi


	mulx	rbx,rsi,r14
	adox	r11,rsi
	adcx	r12,rbx
	mulx	r13,rsi,r15
	mov	rdx,r14
	adox	r12,rsi
	adcx	r13,rdi


	mulx	rbp,rsi,r15
	mov	rdx,r8
	adox	r13,rsi
	adcx	rbp,rdi
	adox	rbp,rdi


	mulx	rsi,r8,rdx
	mov	rdx,rcx
	adcx	r9,r9
	adox	r9,rsi
	adcx	r10,r10
	mulx	rbx,rsi,rdx
	mov	rdx,r14
	adcx	r11,r11
	adox	r10,rsi
	adcx	r12,r12
	adox	r11,rbx
	mulx	rbx,rsi,rdx
	mov	rdx,r15
	adcx	r13,r13
	adox	r12,rsi
	adcx	rbp,rbp
	adox	r13,rbx
	mulx	r15,r14,rdx
	mov	edx,189
	adox	r14,rbp
	adcx	r15,rdi
	adox	r15,rdi


	mulx	rbx,rsi,r12
	adcx	r8,rsi
	adox	r9,rbx
	mulx	rbx,rcx,r13
	adcx	rcx,r9
	adox	r10,rbx
	mulx	rbx,r14,r14
	adcx	r14,r10
	adox	r11,rbx
	mulx	r12,r15,r15
	adcx	r15,r11
	adox	r12,rdi
	adcx	r12,rdi

	mov	eax,eax
	mov	rsi,QWORD PTR[8+rsp]
	imul	rdx,r12

	add	rdx,r8
	adc	rcx,0
	lea	r8,QWORD PTR[189+rdx]
	adc	r14,0
	adc	r15,0

	dec	eax
	jnz	$L$oop_sqrx

	mov	rbp,rdx
	mov	rdx,QWORD PTR[rsi]
	cmovc	rbp,r8

	jmp	$L$mulx_data_is_loaded

$L$SEH_end_sqrx_n_mul_mod_256_189::
sqrx_n_mul_mod_256_189	ENDP

PUBLIC	mulx_mod_256_189


ALIGN	32
mulx_mod_256_189	PROC PUBLIC
	DB	243,15,30,250
	mov	QWORD PTR[8+rsp],rdi	;WIN64 prologue
	mov	QWORD PTR[16+rsp],rsi
	mov	rax,rsp
$L$SEH_begin_mulx_mod_256_189::
	mov	rdi,rcx
	mov	rsi,rdx
	mov	rdx,r8



	push	rbp

	push	rbx

	push	r12

	push	r13

	push	r14

	push	r15

	push	rdi

	lea	rsp,QWORD PTR[((-16))+rsp]


	mov	rax,rdx
	mov	rbp,QWORD PTR[rdx]
	mov	rdx,QWORD PTR[rsi]
	mov	rcx,QWORD PTR[8+rax]
	mov	r14,QWORD PTR[16+rax]
	mov	r15,QWORD PTR[24+rax]

$L$mulx_data_is_loaded::
	mulx	rax,r8,rbp
	xor	edi,edi
	mulx	rbx,r9,rcx
	adcx	r9,rax
	mulx	rax,r10,r14
	adcx	r10,rbx
	mulx	r12,r11,r15
	mov	rdx,QWORD PTR[8+rsi]
	adcx	r11,rax
	mov	QWORD PTR[rsp],r14
	adcx	r12,rdi

	mulx	rbx,rax,rbp
	adox	r9,rax
	adcx	r10,rbx
	mulx	rbx,rax,rcx
	adox	r10,rax
	adcx	r11,rbx
	mulx	rbx,rax,r14
	adox	r11,rax
	adcx	r12,rbx
	mulx	r13,rax,r15
	mov	rdx,QWORD PTR[16+rsi]
	adox	r12,rax
	adcx	r13,rdi
	adox	r13,rdi

	mulx	rbx,rax,rbp
	adcx	r10,rax
	adox	r11,rbx
	mulx	rbx,rax,rcx
	adcx	r11,rax
	adox	r12,rbx
	mulx	rbx,rax,r14
	adcx	r12,rax
	adox	r13,rbx
	mulx	r14,rax,r15
	mov	rdx,QWORD PTR[24+rsi]
	adcx	r13,rax
	adox	r14,rdi
	adcx	r14,rdi

	mulx	rbx,rax,rbp
	adox	r11,rax
	adcx	r12,rbx
	mulx	rbx,rax,rcx
	adox	r12,rax
	adcx	r13,rbx
	mulx	rbx,rax,QWORD PTR[rsp]
	adox	r13,rax
	adcx	r14,rbx
	mulx	r15,rax,r15
	mov	edx,189
	adox	r14,rax
	adcx	r15,rdi
	adox	r15,rdi

	jmp	$L$reduce64

$L$SEH_end_mulx_mod_256_189::
mulx_mod_256_189	ENDP

PUBLIC	sqrx_mod_256_189


ALIGN	32
sqrx_mod_256_189	PROC PUBLIC
	DB	243,15,30,250
	mov	QWORD PTR[8+rsp],rdi	;WIN64 prologue
	mov	QWORD PTR[16+rsp],rsi
	mov	rax,rsp
$L$SEH_begin_sqrx_mod_256_189::
	mov	rdi,rcx
	mov	rsi,rdx



	push	rbp

	push	rbx

	push	r12

	push	r13

	push	r14

	push	r15

	push	rdi

	lea	rsp,QWORD PTR[((-16))+rsp]


	mov	rdx,QWORD PTR[rsi]
	mov	rcx,QWORD PTR[8+rsi]
	mov	rbp,QWORD PTR[16+rsi]
	mov	rsi,QWORD PTR[24+rsi]


	mulx	r15,r8,rdx
	mulx	rax,r9,rcx
	xor	edi,edi
	mulx	rbx,r10,rbp
	adcx	r10,rax
	mulx	r12,r11,rsi
	mov	rdx,rcx
	adcx	r11,rbx
	adcx	r12,rdi


	mulx	rbx,rax,rbp
	adox	r11,rax
	adcx	r12,rbx
	mulx	r13,rax,rsi
	mov	rdx,rbp
	adox	r12,rax
	adcx	r13,rdi


	mulx	r14,rax,rsi
	mov	rdx,rcx
	adox	r13,rax
	adcx	r14,rdi
	adox	r14,rdi

	adcx	r9,r9
	adox	r9,r15
	adcx	r10,r10
	mulx	rbx,rax,rdx
	mov	rdx,rbp
	adcx	r11,r11
	adox	r10,rax
	adcx	r12,r12
	adox	r11,rbx
	mulx	rbx,rax,rdx
	mov	rdx,rsi
	adcx	r13,r13
	adox	r12,rax
	adcx	r14,r14
	adox	r13,rbx
	mulx	r15,rax,rdx
	mov	edx,189
	adox	r14,rax
	adcx	r15,rdi
	adox	r15,rdi
	jmp	$L$reduce64

ALIGN	32
$L$reduce64::
	mulx	rbx,rax,r12
	adcx	r8,rax
	adox	r9,rbx
	mulx	rbx,rax,r13
	adcx	r9,rax
	adox	r10,rbx
	mulx	rbx,rax,r14
	adcx	r10,rax
	adox	r11,rbx
	mulx	r12,rax,r15
	adcx	r11,rax
	adox	r12,rdi
	adcx	r12,rdi

	mov	rdi,QWORD PTR[16+rsp]
	imul	r12,rdx

	add	r8,r12
	adc	r9,0
	adc	r10,0
	adc	r11,0

	lea	rax,QWORD PTR[189+r8]
	cmovc	r8,rax

	mov	QWORD PTR[8+rdi],r9
	mov	QWORD PTR[16+rdi],r10
	mov	QWORD PTR[24+rdi],r11
	mov	QWORD PTR[rdi],r8

	mov	r15,QWORD PTR[24+rsp]

	mov	r14,QWORD PTR[32+rsp]

	mov	r13,QWORD PTR[40+rsp]

	mov	r12,QWORD PTR[48+rsp]

	mov	rbx,QWORD PTR[56+rsp]

	mov	rbp,QWORD PTR[64+rsp]

	lea	rsp,QWORD PTR[72+rsp]

	mov	rdi,QWORD PTR[8+rsp]	;WIN64 epilogue
	mov	rsi,QWORD PTR[16+rsp]
	DB	0F3h,0C3h		;repret

$L$SEH_end_sqrx_mod_256_189::
sqrx_mod_256_189	ENDP

PUBLIC	redc_mod_256_189


ALIGN	32
redc_mod_256_189	PROC PUBLIC
	DB	243,15,30,250
	mov	QWORD PTR[8+rsp],rdi	;WIN64 prologue
	mov	QWORD PTR[16+rsp],rsi
	mov	rax,rsp
$L$SEH_begin_redc_mod_256_189::
	mov	rdi,rcx
	mov	rsi,rdx


	mov	r8,QWORD PTR[rsi]
	mov	r9,QWORD PTR[8+rsi]
	mov	r10,QWORD PTR[16+rsi]
	mov	r11,QWORD PTR[24+rsi]

	mov	rax,r8
	mov	rdx,r9
	mov	rcx,r10
	mov	rsi,r11

	add	r8,189
	adc	r9,0
	adc	r10,0
	adc	r11,0

	cmovnc	r8,rax
	cmovnc	r9,rdx
	cmovnc	r10,rcx
	cmovnc	r11,rsi

	mov	QWORD PTR[rdi],r8
	mov	QWORD PTR[8+rdi],r9
	mov	QWORD PTR[16+rdi],r10
	mov	QWORD PTR[24+rdi],r11

	mov	rdi,QWORD PTR[8+rsp]	;WIN64 epilogue
	mov	rsi,QWORD PTR[16+rsp]
	DB	0F3h,0C3h		;repret
$L$SEH_end_redc_mod_256_189::
redc_mod_256_189	ENDP
PUBLIC	cneg_mod_256_189


ALIGN	32
cneg_mod_256_189	PROC PUBLIC
	DB	243,15,30,250
	mov	rax,QWORD PTR[rdx]
	mov	r10,QWORD PTR[8+rdx]
	mov	r9,QWORD PTR[16+rdx]
	mov	r11,rax
	mov	rdx,QWORD PTR[24+rdx]
	or	rax,r10
	or	rax,r9
	neg	r8
	or	rax,rdx
	cmovnz	rax,r8

	mov	r8,rax
	and	rax,189

	add	r11,rax
	adc	r10,0
	adc	r9,0
	adc	rdx,0

	and	rax,1
	xor	r11,r8
	xor	r10,r8
	xor	r9,r8
	xor	rdx,r8
	add	r11,rax
	adc	r10,0
	adc	r9,0
	adc	rdx,0

	mov	QWORD PTR[rcx],r11
	mov	QWORD PTR[8+rcx],r10
	mov	QWORD PTR[16+rcx],r9
	mov	QWORD PTR[24+rcx],rdx

	DB	0F3h,0C3h		;repret
cneg_mod_256_189	ENDP

PUBLIC	xor_n_check_mod_256_189


ALIGN	32
xor_n_check_mod_256_189	PROC PUBLIC
	DB	243,15,30,250
	mov	r11,QWORD PTR[rdx]
	mov	r10,QWORD PTR[8+rdx]
	mov	r9,QWORD PTR[16+rdx]
	mov	rdx,QWORD PTR[24+rdx]

	xor	r11,QWORD PTR[r8]
	xor	r10,QWORD PTR[8+r8]
	xor	r9,QWORD PTR[16+r8]
	xor	rdx,QWORD PTR[24+r8]

	xor	eax,eax
	mov	QWORD PTR[rcx],r11
	add	r11,189
	mov	QWORD PTR[8+rcx],r10
	adc	r10,0
	mov	QWORD PTR[16+rcx],r9
	adc	r9,0
	mov	QWORD PTR[24+rcx],rdx
	adc	rdx,0
	adc	rax,0

	DB	0F3h,0C3h		;repret
xor_n_check_mod_256_189	ENDP

PUBLIC	swap_neigh_256_189


ALIGN	32
swap_neigh_256_189	PROC PUBLIC
	DB	243,15,30,250
	mov	rax,QWORD PTR[rdx]
	mov	r10,QWORD PTR[8+rdx]
	mov	r9,QWORD PTR[16+rdx]
	mov	r11,rax
	and	rax,1
	mov	rdx,QWORD PTR[24+rdx]

	sub	r11,1
	sbb	r10,0
	lea	rax,QWORD PTR[rax*1+rax]
	sbb	r9,0
	sbb	rdx,0

	add	r11,rax
	adc	r10,0
	adc	r9,0
	adc	rdx,0

	mov	QWORD PTR[rcx],r11
	mov	QWORD PTR[8+rcx],r10
	mov	QWORD PTR[16+rcx],r9
	mov	QWORD PTR[24+rcx],rdx

	DB	0F3h,0C3h		;repret
swap_neigh_256_189	ENDP
.text$	ENDS
.pdata	SEGMENT READONLY ALIGN(4)
ALIGN	4
.pdata	ENDS
.xdata	SEGMENT READONLY ALIGN(8)
ALIGN	8

.xdata	ENDS
END
