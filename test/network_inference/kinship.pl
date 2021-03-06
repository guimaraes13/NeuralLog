male(alfonso).
male(andrew).
male(arthur).
male(charles).
male(christopher).
male(colin).
male(emilio).
male(james).
male(marco).
male(pierro).
male(roberto).
male(tomaso).

female(angela).
female(charlotte).
female(christine).
female(francesca).
female(gina).
female(jennifer).
female(lucia).
female(margaret).
female(maria).
female(penelope).
female(sophia).
female(victoria).

age(alfonso, 2).
age(andrew, 67).
age(angela, 5).
age(arthur, 37).
age(charlotte, 11).
age(christine, 13).
age(christopher, 59).
age(colin, 19).
age(emilio, 23).
age(francesca, 29).
age(james, 61).
age(jennifer, 3).
age(lucia, 41).
age(marco, 53).
age(maria, 3).
age(penelope, 43).
age(pierro, 71).
age(roberto, 31).
age(sophia, 47).
age(victoria, 17).

father(andrew, jennifer).
father(christopher, victoria).
father(james, victoria).
father(marco, sophia).
father(pierro, angela).
father(roberto, emilio).
father(roberto, lucia).

mother(christine, jennifer).
mother(francesca, angela).
mother(lucia, sophia).
mother(maria, emilio).
mother(maria, lucia).
mother(penelope, victoria).
mother(victoria, charlotte).

son(james, andrew).
son(arthur, christopher).
son(colin, james).
son(alfonso, marco).
son(marco, pierro).
son(emilio, roberto).

father(X, Y) :- son(Y, X), male(X).

parent(X, Y) :- mother(X, Y).
parent(X, Y) :- father(X, Y).

grand_mother(X, Y) :- mother(X, lucia), parent(lucia, Y).
grand_grand_father(X, Y) :- father(X, james), parent(james, victoria),
    parent(victoria, Y).

avgAgeFriends(X) :- father(X, Z), age(Z, Y), mean(Y).
ageFriends(X, Y) :- father(X, Z), age(Z, Y).

1.732::l_1(andrew).
1.0::l_2(andrew).

1.4142::l_1(christopher).
1.4142::l_2(christopher).

1.0::l_1(james).
1.732::l_2(james).

h_1(X, Y) :- l_1(X), l_1(Y). %% X_1 * Y_1
h_2(X, Y) :- l_2(X), l_2(Y). %% X_2 * Y_2

num(X, Y) :- h_1(X, Y).  %% (X_1 * Y_1) + (X_2 * Y_2)
num(X, Y) :- h_2(X, Y).

square_sum(X) :- l_1(X), l_1(X).  %% X_1^2 + X_2^2
square_sum(X) :- l_2(X), l_2(X).

norm(X) :- square_sum(X), square_root(X).

den(X, Y) :- norm(X), norm(Y).
inv_den(X, Y) :- den(X, Y), inverse(Y).

similarity(X, Y) :- num(X, Y), inv_den(X, Y).


h_1(X, Y, Z) :- l_1(X), l_1(Y), l_1(Z).
h_2(X, Y, Z) :- l_2(X), l_2(Y), l_2(Z).

num(X, Y, Z) :- h_1(X, Y, Z).
num(X, Y, Z) :- h_2(X, Y, Z).

den(X, Y, Z) :- norm(X), norm(Y), norm(Z).
inv_den(X, Y, Z) :- den(X, Y, Z), inverse(Z).

similarity(X, Y, Z) :- num(X, Y, Z), inv_den(X, Y, Z).

parents(X, Y, Z) :- mother(X, Z), father(Y, Z).

parents(X, Y) :- parents(X, C, Y).

wrong_x(X, Y) :- father(X, X).
wrong_y(X, Y) :- mother(Y, Y).
