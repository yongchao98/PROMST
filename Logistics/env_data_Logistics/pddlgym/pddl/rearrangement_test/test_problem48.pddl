(define (problem rearrangement) 
    (:domain rearrangement)

    (:objects
    
	pawn-0 - moveable
	pawn-1 - moveable
	pawn-2 - moveable
	bear-3 - moveable
	robot - moveable
	loc-0-0 - static
	loc-0-1 - static
	loc-0-2 - static
	loc-0-3 - static
	loc-0-4 - static
	loc-1-0 - static
	loc-1-1 - static
	loc-1-2 - static
	loc-1-3 - static
	loc-1-4 - static
	loc-2-0 - static
	loc-2-1 - static
	loc-2-2 - static
	loc-2-3 - static
	loc-2-4 - static
	loc-3-0 - static
	loc-3-1 - static
	loc-3-2 - static
	loc-3-3 - static
	loc-3-4 - static
    )

    (:init
    
	(ispawn pawn-0)
	(ispawn pawn-1)
	(ispawn pawn-2)
	(isbear bear-3)
	(isrobot robot)
	(at pawn-0 loc-3-3)
	(at pawn-1 loc-2-2)
	(at pawn-2 loc-3-2)
	(at bear-3 loc-2-1)
	(at robot loc-3-2)
	(handsfree robot)

    ; action literals
    
	(pick pawn-0)
	(place pawn-0)
	(pick pawn-1)
	(place pawn-1)
	(pick pawn-2)
	(place pawn-2)
	(pick bear-3)
	(place bear-3)
	(moveto loc-0-0)
	(moveto loc-0-1)
	(moveto loc-0-2)
	(moveto loc-0-3)
	(moveto loc-0-4)
	(moveto loc-1-0)
	(moveto loc-1-1)
	(moveto loc-1-2)
	(moveto loc-1-3)
	(moveto loc-1-4)
	(moveto loc-2-0)
	(moveto loc-2-1)
	(moveto loc-2-2)
	(moveto loc-2-3)
	(moveto loc-2-4)
	(moveto loc-3-0)
	(moveto loc-3-1)
	(moveto loc-3-2)
	(moveto loc-3-3)
	(moveto loc-3-4)
    )

    (:goal (and  (at pawn-0 loc-1-0) ))
)
    