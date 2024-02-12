(define (problem doors) 
    (:domain glibdoors)

    (:objects
    key-0 - key
	key-1 - key
	key-2 - key
	key-3 - key
	loc-0-0 - location
	loc-0-1 - location
	loc-0-2 - location
	loc-0-3 - location
	loc-0-4 - location
	loc-0-5 - location
	loc-0-6 - location
	loc-0-7 - location
	loc-0-8 - location
	loc-0-9 - location
	loc-1-0 - location
	loc-1-1 - location
	loc-1-2 - location
	loc-1-3 - location
	loc-1-4 - location
	loc-1-5 - location
	loc-1-6 - location
	loc-1-7 - location
	loc-1-8 - location
	loc-1-9 - location
	loc-2-0 - location
	loc-2-1 - location
	loc-2-2 - location
	loc-2-3 - location
	loc-2-4 - location
	loc-2-5 - location
	loc-2-6 - location
	loc-2-7 - location
	loc-2-8 - location
	loc-2-9 - location
	loc-3-0 - location
	loc-3-1 - location
	loc-3-2 - location
	loc-3-3 - location
	loc-3-4 - location
	loc-3-5 - location
	loc-3-6 - location
	loc-3-7 - location
	loc-3-8 - location
	loc-3-9 - location
	loc-4-0 - location
	loc-4-1 - location
	loc-4-2 - location
	loc-4-3 - location
	loc-4-4 - location
	loc-4-5 - location
	loc-4-6 - location
	loc-4-7 - location
	loc-4-8 - location
	loc-4-9 - location
	loc-5-0 - location
	loc-5-1 - location
	loc-5-2 - location
	loc-5-3 - location
	loc-5-4 - location
	loc-5-5 - location
	loc-5-6 - location
	loc-5-7 - location
	loc-5-8 - location
	loc-5-9 - location
	loc-6-0 - location
	loc-6-1 - location
	loc-6-2 - location
	loc-6-3 - location
	loc-6-4 - location
	loc-6-5 - location
	loc-6-6 - location
	loc-6-7 - location
	loc-6-8 - location
	loc-6-9 - location
	loc-7-0 - location
	loc-7-1 - location
	loc-7-2 - location
	loc-7-3 - location
	loc-7-4 - location
	loc-7-5 - location
	loc-7-6 - location
	loc-7-7 - location
	loc-7-8 - location
	loc-7-9 - location
	room-0 - room
	room-1 - room
	room-2 - room
	room-3 - room
	room-4 - room
	room-5 - room
    )

    (:init
    (at loc-0-0)
	(unlocked room-0)
	(locinroom loc-0-0 room-0)
	(moveto loc-0-0)
	(locinroom loc-0-1 room-1)
	(moveto loc-0-1)
	(locinroom loc-0-2 room-2)
	(moveto loc-0-2)
	(locinroom loc-0-3 room-3)
	(moveto loc-0-3)
	(locinroom loc-0-4 room-4)
	(moveto loc-0-4)
	(locinroom loc-0-5 room-0)
	(moveto loc-0-5)
	(locinroom loc-0-6 room-0)
	(moveto loc-0-6)
	(locinroom loc-0-7 room-0)
	(moveto loc-0-7)
	(locinroom loc-0-8 room-0)
	(moveto loc-0-8)
	(locinroom loc-0-9 room-0)
	(moveto loc-0-9)
	(locinroom loc-1-0 room-0)
	(moveto loc-1-0)
	(locinroom loc-1-1 room-0)
	(moveto loc-1-1)
	(locinroom loc-1-2 room-0)
	(moveto loc-1-2)
	(locinroom loc-1-3 room-0)
	(moveto loc-1-3)
	(locinroom loc-1-4 room-4)
	(moveto loc-1-4)
	(locinroom loc-1-5 room-0)
	(moveto loc-1-5)
	(locinroom loc-1-6 room-0)
	(moveto loc-1-6)
	(locinroom loc-1-7 room-0)
	(moveto loc-1-7)
	(locinroom loc-1-8 room-0)
	(moveto loc-1-8)
	(locinroom loc-1-9 room-0)
	(moveto loc-1-9)
	(locinroom loc-2-0 room-0)
	(moveto loc-2-0)
	(locinroom loc-2-1 room-0)
	(moveto loc-2-1)
	(locinroom loc-2-2 room-0)
	(moveto loc-2-2)
	(locinroom loc-2-3 room-0)
	(moveto loc-2-3)
	(locinroom loc-2-4 room-4)
	(moveto loc-2-4)
	(locinroom loc-2-5 room-0)
	(moveto loc-2-5)
	(locinroom loc-2-6 room-0)
	(moveto loc-2-6)
	(locinroom loc-2-7 room-0)
	(moveto loc-2-7)
	(locinroom loc-2-8 room-0)
	(moveto loc-2-8)
	(locinroom loc-2-9 room-0)
	(moveto loc-2-9)
	(locinroom loc-3-0 room-0)
	(moveto loc-3-0)
	(locinroom loc-3-1 room-0)
	(moveto loc-3-1)
	(locinroom loc-3-2 room-0)
	(moveto loc-3-2)
	(locinroom loc-3-3 room-0)
	(moveto loc-3-3)
	(locinroom loc-3-4 room-0)
	(moveto loc-3-4)
	(locinroom loc-3-5 room-0)
	(moveto loc-3-5)
	(locinroom loc-3-6 room-0)
	(moveto loc-3-6)
	(locinroom loc-3-7 room-0)
	(moveto loc-3-7)
	(locinroom loc-3-8 room-0)
	(moveto loc-3-8)
	(locinroom loc-3-9 room-0)
	(moveto loc-3-9)
	(locinroom loc-4-0 room-0)
	(moveto loc-4-0)
	(locinroom loc-4-1 room-0)
	(moveto loc-4-1)
	(locinroom loc-4-2 room-0)
	(moveto loc-4-2)
	(locinroom loc-4-3 room-0)
	(moveto loc-4-3)
	(locinroom loc-4-4 room-0)
	(moveto loc-4-4)
	(locinroom loc-4-5 room-0)
	(moveto loc-4-5)
	(locinroom loc-4-6 room-0)
	(moveto loc-4-6)
	(locinroom loc-4-7 room-0)
	(moveto loc-4-7)
	(locinroom loc-4-8 room-0)
	(moveto loc-4-8)
	(locinroom loc-4-9 room-0)
	(moveto loc-4-9)
	(locinroom loc-5-0 room-0)
	(moveto loc-5-0)
	(locinroom loc-5-1 room-0)
	(moveto loc-5-1)
	(locinroom loc-5-2 room-0)
	(moveto loc-5-2)
	(locinroom loc-5-3 room-0)
	(moveto loc-5-3)
	(locinroom loc-5-4 room-0)
	(moveto loc-5-4)
	(locinroom loc-5-5 room-0)
	(moveto loc-5-5)
	(locinroom loc-5-6 room-0)
	(moveto loc-5-6)
	(locinroom loc-5-7 room-0)
	(moveto loc-5-7)
	(locinroom loc-5-8 room-0)
	(moveto loc-5-8)
	(locinroom loc-5-9 room-0)
	(moveto loc-5-9)
	(locinroom loc-6-0 room-0)
	(moveto loc-6-0)
	(locinroom loc-6-1 room-0)
	(moveto loc-6-1)
	(locinroom loc-6-2 room-0)
	(moveto loc-6-2)
	(locinroom loc-6-3 room-0)
	(moveto loc-6-3)
	(locinroom loc-6-4 room-0)
	(moveto loc-6-4)
	(locinroom loc-6-5 room-0)
	(moveto loc-6-5)
	(locinroom loc-6-6 room-0)
	(moveto loc-6-6)
	(locinroom loc-6-7 room-0)
	(moveto loc-6-7)
	(locinroom loc-6-8 room-0)
	(moveto loc-6-8)
	(locinroom loc-6-9 room-0)
	(moveto loc-6-9)
	(locinroom loc-7-0 room-0)
	(moveto loc-7-0)
	(locinroom loc-7-1 room-0)
	(moveto loc-7-1)
	(locinroom loc-7-2 room-0)
	(moveto loc-7-2)
	(locinroom loc-7-3 room-0)
	(moveto loc-7-3)
	(locinroom loc-7-4 room-0)
	(moveto loc-7-4)
	(locinroom loc-7-5 room-0)
	(moveto loc-7-5)
	(locinroom loc-7-6 room-0)
	(moveto loc-7-6)
	(locinroom loc-7-7 room-0)
	(moveto loc-7-7)
	(locinroom loc-7-8 room-0)
	(moveto loc-7-8)
	(locinroom loc-7-9 room-0)
	(moveto loc-7-9)
	(keyforroom key-0 room-1)
	(keyat key-0 loc-3-3)
	(pick key-0)
	(keyforroom key-1 room-5)
	(keyat key-1 loc-0-4)
	(pick key-1)
	(keyforroom key-2 room-2)
	(keyat key-2 loc-0-1)
	(pick key-2)
	(keyforroom key-3 room-4)
	(keyat key-3 loc-0-3)
	(pick key-3)

    )

    (:goal (and (at loc-0-4)))
)
    