(define (problem minecraft) 
    (:domain minecraft)

    (:objects
    
	grass-0 - moveable
	grass-1 - moveable
	log-2 - moveable
	new-0 - moveable
	new-1 - moveable
	new-2 - moveable
	agent - agent
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
    
	(hypothetical new-0)
	(hypothetical new-1)
	(hypothetical new-2)
	(isgrass grass-0)
	(isgrass grass-1)
	(islog log-2)
	(at grass-0 loc-1-2)
	(at grass-1 loc-1-2)
	(at log-2 loc-0-2)
	(agentat loc-1-1)
	(handsfree agent)

    ; action literals
    
	(recall grass-0)
	(craftplank grass-0 grass-1)
	(craftplank grass-0 log-2)
	(craftplank grass-0 new-0)
	(craftplank grass-0 new-1)
	(craftplank grass-0 new-2)
	(equip grass-0)
	(pick grass-0)
	(recall grass-1)
	(craftplank grass-1 grass-0)
	(craftplank grass-1 log-2)
	(craftplank grass-1 new-0)
	(craftplank grass-1 new-1)
	(craftplank grass-1 new-2)
	(equip grass-1)
	(pick grass-1)
	(recall log-2)
	(craftplank log-2 grass-0)
	(craftplank log-2 grass-1)
	(craftplank log-2 new-0)
	(craftplank log-2 new-1)
	(craftplank log-2 new-2)
	(equip log-2)
	(pick log-2)
	(recall new-0)
	(craftplank new-0 grass-0)
	(craftplank new-0 grass-1)
	(craftplank new-0 log-2)
	(craftplank new-0 new-1)
	(craftplank new-0 new-2)
	(equip new-0)
	(pick new-0)
	(recall new-1)
	(craftplank new-1 grass-0)
	(craftplank new-1 grass-1)
	(craftplank new-1 log-2)
	(craftplank new-1 new-0)
	(craftplank new-1 new-2)
	(equip new-1)
	(pick new-1)
	(recall new-2)
	(craftplank new-2 grass-0)
	(craftplank new-2 grass-1)
	(craftplank new-2 log-2)
	(craftplank new-2 new-0)
	(craftplank new-2 new-1)
	(equip new-2)
	(pick new-2)
	(move loc-0-0)
	(move loc-0-1)
	(move loc-0-2)
	(move loc-0-3)
	(move loc-0-4)
	(move loc-1-0)
	(move loc-1-1)
	(move loc-1-2)
	(move loc-1-3)
	(move loc-1-4)
	(move loc-2-0)
	(move loc-2-1)
	(move loc-2-2)
	(move loc-2-3)
	(move loc-2-4)
	(move loc-3-0)
	(move loc-3-1)
	(move loc-3-2)
	(move loc-3-3)
	(move loc-3-4)
    )

    (:goal (and  (agentat loc-1-0)  (inventory new-0) ))
)
    