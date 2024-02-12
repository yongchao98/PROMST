(define (problem minecraft) 
    (:domain minecraft)

    (:objects
    
	log-0 - moveable
	log-1 - moveable
	grass-2 - moveable
	log-3 - moveable
	new-0 - moveable
	new-1 - moveable
	new-2 - moveable
	agent - agent
	loc-0-0 - static
	loc-0-1 - static
	loc-0-2 - static
	loc-0-3 - static
	loc-1-0 - static
	loc-1-1 - static
	loc-1-2 - static
	loc-1-3 - static
	loc-2-0 - static
	loc-2-1 - static
	loc-2-2 - static
	loc-2-3 - static
	loc-3-0 - static
	loc-3-1 - static
	loc-3-2 - static
	loc-3-3 - static
    )

    (:init
    
	(hypothetical new-0)
	(hypothetical new-1)
	(hypothetical new-2)
	(islog log-0)
	(islog log-1)
	(isgrass grass-2)
	(islog log-3)
	(at log-0 loc-3-0)
	(at log-1 loc-0-3)
	(at grass-2 loc-0-2)
	(at log-3 loc-2-0)
	(agentat loc-2-1)
	(handsfree agent)

    ; action literals
    
	(recall log-0)
	(craftplank log-0 log-1)
	(craftplank log-0 grass-2)
	(craftplank log-0 log-3)
	(craftplank log-0 new-0)
	(craftplank log-0 new-1)
	(craftplank log-0 new-2)
	(equip log-0)
	(pick log-0)
	(recall log-1)
	(craftplank log-1 log-0)
	(craftplank log-1 grass-2)
	(craftplank log-1 log-3)
	(craftplank log-1 new-0)
	(craftplank log-1 new-1)
	(craftplank log-1 new-2)
	(equip log-1)
	(pick log-1)
	(recall grass-2)
	(craftplank grass-2 log-0)
	(craftplank grass-2 log-1)
	(craftplank grass-2 log-3)
	(craftplank grass-2 new-0)
	(craftplank grass-2 new-1)
	(craftplank grass-2 new-2)
	(equip grass-2)
	(pick grass-2)
	(recall log-3)
	(craftplank log-3 log-0)
	(craftplank log-3 log-1)
	(craftplank log-3 grass-2)
	(craftplank log-3 new-0)
	(craftplank log-3 new-1)
	(craftplank log-3 new-2)
	(equip log-3)
	(pick log-3)
	(recall new-0)
	(craftplank new-0 log-0)
	(craftplank new-0 log-1)
	(craftplank new-0 grass-2)
	(craftplank new-0 log-3)
	(craftplank new-0 new-1)
	(craftplank new-0 new-2)
	(equip new-0)
	(pick new-0)
	(recall new-1)
	(craftplank new-1 log-0)
	(craftplank new-1 log-1)
	(craftplank new-1 grass-2)
	(craftplank new-1 log-3)
	(craftplank new-1 new-0)
	(craftplank new-1 new-2)
	(equip new-1)
	(pick new-1)
	(recall new-2)
	(craftplank new-2 log-0)
	(craftplank new-2 log-1)
	(craftplank new-2 grass-2)
	(craftplank new-2 log-3)
	(craftplank new-2 new-0)
	(craftplank new-2 new-1)
	(equip new-2)
	(pick new-2)
	(move loc-0-0)
	(move loc-0-1)
	(move loc-0-2)
	(move loc-0-3)
	(move loc-1-0)
	(move loc-1-1)
	(move loc-1-2)
	(move loc-1-3)
	(move loc-2-0)
	(move loc-2-1)
	(move loc-2-2)
	(move loc-2-3)
	(move loc-3-0)
	(move loc-3-1)
	(move loc-3-2)
	(move loc-3-3)
    )

    (:goal (and  (agentat loc-3-1)  (inventory log-1) ))
)
    