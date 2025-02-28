## CTMC Solver

A quality tester is fed by material flows from two different sources. Only one of the two sources 
can be active, either source 0 or source 1. The average duration of an activity period of source 0 is 
2 minutes. The average duration of source 1 being active is 3 minutes. At the beginning of the 
simulation source 0 is active. The code solves the CTMC of this system, by DTMC discretization. 

Note : In each time step, one item is produced, the probability for the item to test OK is 
0.9 for source 0 and 0.95 for source 1. 
