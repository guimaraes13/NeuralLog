parent(X0, X1) :- father(X0, X1), false.
parent(X0, X1) :- father(X1, X0).
