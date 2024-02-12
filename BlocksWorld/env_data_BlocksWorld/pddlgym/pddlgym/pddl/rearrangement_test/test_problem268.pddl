(define (problem rearrangement) 
    (:domain rearrangement)

    (:objects
    
	monkey-0 - moveable
	robot - moveable
	loc-0-0 - static
	loc-0-1 - static
	loc-0-2 - static
	loc-1-0 - static
	loc-1-1 - static
	loc-1-2 - static
	loc-2-0 - static
	loc-2-1 - static
	loc-2-2 - static
    )

    (:init
    
	(ismonkey monkey-0)
	(isrobot robot)
	(at monkey-0 loc-2-0)
	(at robot loc-1-0)
	(handsfree robot)

    ; action literals
    
	(pick monkey-0)
	(place monkey-0)
	(moveto loc-0-0)
	(moveto loc-0-1)
	(moveto loc-0-2)
	(moveto loc-1-0)
	(moveto loc-1-1)
	(moveto loc-1-2)
	(moveto loc-2-0)
	(moveto loc-2-1)
	(moveto loc-2-2)
    )

    (:goal (and  (at monkey-0 loc-0-0) ))
)
    