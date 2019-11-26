father(andrew, james).
father(andrew, jennifer).
father(arthur, colin).
father(christopher, arthur).
father(christopher, victoria).
father(james, charlotte).
father(james, colin).
father(marco, alfonso).
father(alfonso, lucia).
father(marco, sophia).
father(pierro, angela).
father(pierro, marco).
father(roberto, emilio).
father(roberto, lucia).

mother(christine, jennifer).
mother(francesca, angela).
mother(francesca, marco).
mother(lucia, sophia).
mother(lucia, angela).
mother(maria, emilio).
mother(maria, lucia).
mother(maria, francesca).
mother(penelope, arthur).
mother(angela, arthur).
mother(angela, victoria).
mother(angela, emilio).
mother(victoria, charlotte).
mother(victoria, emilio).

parent(X, Y) :- father(X, Y).
parent(X, Y) :- mother(X, Y).

grand_father(X, Y) :- father(X, Z), parent(Z, Y), w_f(X).
grand_mother(X, Y) :- mother(X, Z), parent(Z, Y), w_m(X).
