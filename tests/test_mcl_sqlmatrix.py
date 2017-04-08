import graph
import mcl_sqlmatrix
import matrix
import sqlalchemy

def compare_clusters(A,B):
    if len(A) != len(B):
        return False
    for x in A:
        for y in B:
            if set(x)==set(y):
                # We've found a cluster in B that matches this cluster in A
                # Go to the next cluster in B.
                break
        else:
            #We went all the way through B without finding a match
            return False
    return True


if __name__=='__main__':
    # Set up the class
    engine = sqlalchemy.create_engine('sqlite:///:memory:')
    matrix.SqlMatrix.setup(engine)
    # Graph from Figure 10 (page 45) of MCL thesis
    G1 = graph.Graph({'A':['B','F','G','J'],
            'B':['A','C','E'],
            'C':['B','D','E'],
            'D':['C','H','I','K'],
            'E':['B','C','G','H'],
            'F':['A','J'],
            'G':['A','E','J'],
            'H':['D','E','I','K'],
            'I':['D','H','K','L'],
            'J':['A','F','G'],
            'K':['D','H','I','L'],
            'L':['I','K']})
    M1,labels1 = G1.matrix()
    SM1 = matrix.SqlMatrix.from_dense(M1)
    E1 = mcl_sqlmatrix.param_iter(tail=2)
    R1 = mcl_sqlmatrix.param_iter(tail=2)
    print(mcl_sqlmatrix.MCL(SM1,E1,R1,labels=labels1))
    # Graph from Figure 5 (page 42) of MCL thesis
    # "Answer" in figure 15 (page 56)
    G2 = graph.Graph({'A':['B','C','D'],
            'B':['A','C','D','E'],
            'C':['A','B','D'],
            'D':['A','B','C','E'],
            'E':['B','D','F'],
            'F':['E','G','H','I','J','L'],
            'G':['F','H','I','J'],
            'H':['F','G','I','J','N','O','P'],
            'I':['F','G','H','J','K'],
            'J':['F','G','H','I'],
            'K':['I','L','P'],
            'L':['F','K','M'],
            'M':['L'],
            'N':['H','O','Q','R'],
            'O':['H','N','P','Q'],
            'P':['H','K','O','Q','R'],
            'Q':['N','O','P','R','S','T'],
            'R':['N','P','Q','S','T'],
            'S':['Q','R','T'],
            'T':['Q','R','S']})
    M2,labels2 = G2.matrix()
    SM2 = matrix.SqlMatrix.from_dense(M2)
    E2 = mcl_sqlmatrix.param_iter(tail=2)
    R2 = mcl_sqlmatrix.param_iter(tail=2)
    print(mcl_sqlmatrix.MCL(SM2,E2,R2,labels=labels2))