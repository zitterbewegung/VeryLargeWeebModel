import snappy

brd=[1,-4,2,3,3,3,2,3,2,2,4,-3,-3,-3,-3,-1,-3,-2,-3,-3]
print('First braid word (for L) is')
print(str(brd))
L=snappy.Link(braid_closure=brd)  ## building the connected sum
L.simplify('global')
print('Summands of L, and their number of crossings:')
PP=K.deconnect_sum()      ## this yields 7_1 and mirror(7_1)
print(str(PP))
print('Attempting to identify summands:')
print(str(PP[0].exterior().identify()))  ## these may be "[]"; SnapPy does
print(str(PP[1].exterior().identify()))  ## not always succeed in 'recognizing'
## non-hyperbolic knots.

print('First summand knot group is ')
print(str(PP[0].exterior().fundamental_group()))
print('Second summand knot group is ')
print(str(PP[1].exterior().fundamental_group()))
## both of these will yield <a,b|aaaaaaabb> (or <a,b|aabbbbbbb>);
## 7_1 is the only knot with this group.

brd2=brd[:]    ## implementing the 2 crossing changes
brd2[0]=-brd2[0]
brd2[1]=-brd2[1]
LA=snappy.Link(braid_closure=brd2)  ## this is K14a18636
print('Second braid word (for LA) is ')
print(str(brd2))
LA.simplify('global')
print('LA is '+str(LA.exterior().identify()))

DTC=[4,-16,24,26,18,20,28,22,-2,10,12,30,6,8,14]
DTCB=DTC[:]
print('DT code for KA is ')
print(str(DTC))
DTCB[0]=-DTCB[0]  ## change one crossing
print('DT code for KB is ')
print(str(DTCB))
KA=snappy.Link('DT:'+str(DTC))    ## this is K14a18636
KB=snappy.Link('DT:'+str(DTCB))   ## this is K15n81556
print('KA is '+str(KA.exterior().identify()))
print('KB is '+str(KB.exterior().identify()))

print('SnapPy check: LA and KA are the same knot? '
+str(LA.exterior().is_isometric_to(KA.exterior())))

DTD=[4,12,-24,14,18,2,20,26,8,10,-28,-30,16,-6,-22]
print('DT code for KC is ')
print(str(DTD))
DTDB=DTD[:]
DTDB[6]=-DTDB[6]  ## change one crossing
print('DT code for KD is ')
print(str(DTDB))
KC=snappy.Link('DT:'+str(DTD))    ## this is K15n81556
KD=snappy.Link('DT:'+str(DTDB))   ## this is K12n412
print('KC is '+str(KC.exterior().identify()))
print('KD is '+str(KD.exterior().identify()))

print('SnapPy check: KB and KC are the same knot? '
+str(KB.exterior().is_isometric_to(KC.exterior())))

DTE=DTDB[:]
DTE[13]=-DTE[13]
print('DT code for KE is ')
print(str(DTE))
KE=snappy.Link('DT:'+str(DTE))    ## this is the unknot
print('KE is '+str(KE.exterior().identify()))

print('Knot group for KE:')
print(str(KE.exterior().fundamental_group()))
## only the unknot has cyclic knot group
