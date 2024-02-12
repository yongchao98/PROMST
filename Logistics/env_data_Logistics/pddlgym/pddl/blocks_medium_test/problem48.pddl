
(define (problem generatedblocks) (:domain blocks)
  (:objects
        b0 - block
	b1 - block
	b10 - block
	b11 - block
	b12 - block
	b13 - block
	b14 - block
	b15 - block
	b16 - block
	b17 - block
	b18 - block
	b19 - block
	b2 - block
	b20 - block
	b21 - block
	b3 - block
	b4 - block
	b5 - block
	b6 - block
	b7 - block
	b8 - block
	b9 - block
  )
  (:init 
	(clear b0)
	(clear b10)
	(clear b14)
	(clear b19)
	(clear b5)
	(handempty)
	(on b0 b1)
	(on b10 b11)
	(on b11 b12)
	(on b12 b13)
	(on b14 b15)
	(on b15 b16)
	(on b16 b17)
	(on b17 b18)
	(on b19 b20)
	(on b1 b2)
	(on b20 b21)
	(on b2 b3)
	(on b3 b4)
	(on b5 b6)
	(on b6 b7)
	(on b7 b8)
	(on b8 b9)
	(ontable b13)
	(ontable b18)
	(ontable b21)
	(ontable b4)
	(ontable b9)
  )
  (:goal (and
	(on b0 b15)
	(on b15 b9)
	(ontable b9)
	(on b19 b1)
	(on b1 b13)
	(ontable b13)
	(on b21 b20)
	(on b20 b18)
	(on b18 b12)
	(on b12 b8)
	(ontable b8)
	(on b14 b4)
	(on b4 b11)
	(on b11 b16)
	(ontable b16)))
)
