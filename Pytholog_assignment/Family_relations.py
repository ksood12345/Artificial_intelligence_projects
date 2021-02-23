import pytholog as pl


kb = pl.KnowledgeBase("family_tree")
kb(['male(abe)', 'male(clancy)', 'female(mona)', 'female(jacqueline)',
    'male(herb)', 'male(homer)', 'female(marge)', 'female(patty)', 'female(selma)',
    'male(bart)', 'female(lisa)', 'female(maggie)', 'male(ling)',
    'father(abe,herb)', 'father(abe,homer)', 'mother(mona,herb)', 'mother(mona,homer)',
    'father(clancy,marge)', 'father(clancy,patty)', 'father(clancy,selma)',
    'mother(jacqueline,marge)', 'mother(jacqueline,patty)', 'mother(jacqueline,selma)',
    'father(homer,bart)', 'father(homer,lisa)', 'father(homer,maggie)',
    'mother(marge,bart)', 'mother(marge,lisa)', 'mother(marge,maggie)', 'mother(selma,ling)',
    'parent(X,Y):- father(X,Y)',
    'parent(X,Y):- mother(X,Y)',
    'brother(X,Y):- male(X), parent(Z,X), parent(Z,Y), neq(X,Y)',
    'sister(X,Y):- female(X), parent(Z,X), parent(Z,Y), neq(X,Y)',
    'grandfather(X,Y):- male(X), parent(X,Z), parent(Z,Y)',
    'grandmother(X,Y):- female(X), parent(X,Z), parent(Z,Y)',
    'grandparent(X,Y):- grandfather(X,Y)',
    'grandparent(X,Y):- grandmother(X,Y)',
    'uncle(X,Y):- male(X), father(Z,Y), brother(Z,X)',
    'aunt(X,Y):- female(X), parent(Z,Y), sister(Z,X)',
    'nephew(X,Y):- uncle(Y,X)', 'nephew(X,Y):- aunt(Y,X)'])

# Querying the database
# The following query helps to find father of 'homer'
print('The father of homer is:'+ str(kb.query(pl.Expr("father(X,homer)"))))

# The following query helps to find mother of 'lisa'
print('The mother of homer is:'+ str(kb.query(pl.Expr("mother(X,homer)"))))

# The following query finds both father and mother of 'bart' who happen to be 'homer' and 'marge' respectively.
print('The parents of bart are:'+ str(kb.query(pl.Expr("parent(X,bart)"))))

# The following query helps to find brother of 'homer'
print('The brother of homer is:'+ str(kb.query(pl.Expr("brother(X,homer)"))))

# The following query helps to find sister of 'lisa'
print('The sister of lisa is:'+ str(kb.query(pl.Expr("sister(X,lisa)"))))

# The following query helps to find aunt of 'lisa'
print('The aunts of lisa are:'+ str(kb.query(pl.Expr("aunt(X,lisa)"))))

# The following query helps to find uncle of 'lisa'
print('The uncles of lisa are:'+ str(kb.query(pl.Expr("uncle(X,lisa)"))))

# The following query helps to find grandmother of 'lisa'
print('The grandmother of lisa is:'+ str(kb.query(pl.Expr("grandmother(X,lisa)"))))

# The following query helps to find grandfather of 'ling'
print('The grandfather of lisa is:'+ str(kb.query(pl.Expr("grandfather(X,lisa)"))))

# The following query helps to find grandparent of 'lisa'. The query returns both maternal and paternal grand parents.
print('The grandparents of lisa are:'+ str(kb.query(pl.Expr("grandparent(X,lisa)"))))
    
# The following code helps us understand who is 'lisa' nicece to.
print('Lisa is niece to following people: ' + str(kb.query(pl.Expr("nephew(lisa,X)"))))




