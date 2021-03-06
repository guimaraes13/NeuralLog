set_parameter(initial_value, class_name, random_uniform).
set_parameter(initial_value, config, minval, 0.02).
set_parameter(initial_value, config, maxval, 0.05).

learn(predicate_0_trainable).
learn("century/1").
learn("female/1").
learn("it_female/1").
learn("multiply_2/2").
learn("height/2").
learn("it_height/2").
learn("inv_age/2").
learn("it_inv_age/2").

learn("uncle/2").
learn("it_uncle/2").
learn("aunt/2").
learn("it_aunt/2").
learn("nephew/2").
learn("niece/2").
learn("it_niece/2").

0.3::predicate_0_not_trainable.
0.5::predicate_0_trainable.

0.7::year(2019).
0.9::century(21).

0.5::multiply(2, 3).

0.25::multiply_2(4, 3).

0.02::male(alfonso).
0.03::male(andrew).
0.05::male(arthur).
male(charles).
0.07::male(christopher).
0.07::male(some_male).
0.11::male(colin).
0.13::male(emilio).
0.17::male(james).
0.19::male(marco).
0.23::male(pierro).
0.31::male(roberto).
0.37::male(tomaso).

0.101::female(angela).
0.103::female(charlotte).
0.107::female(christine).
0.107::female(some_female).
0.109::female(francesca).
0.113::female(gina).
0.127::female(jennifer).
0.137::female(lucia).
0.139::female(margaret).
0.149::female(maria).
0.151::female(penelope).
0.157::female(sophia).
0.163::female(victoria).

0.02::it_male(alfonso).
0.03::it_male(andrew).
0.05::it_male(arthur).
it_male(charles).
0.07::it_male(christopher).
0.07::it_male(some_male).
0.11::it_male(colin).
0.13::it_male(emilio).
0.17::it_male(james).
0.19::it_male(marco).
0.23::it_male(pierro).
0.31::it_male(roberto).
0.37::it_male(tomaso).

0.101::it_female(angela).
0.103::it_female(charlotte).
0.107::it_female(christine).
0.107::it_female(some_female).
0.109::it_female(francesca).
0.113::it_female(gina).
0.127::it_female(jennifer).
0.137::it_female(lucia).
0.139::it_female(margaret).
0.149::it_female(maria).
0.151::it_female(penelope).
0.157::it_female(sophia).
0.163::it_female(victoria).

0.211::age(some_male, 30).
0.223::age(colin, 41).
0.227::age(emilio, 27).
0.239::age(some_female, 23).
0.241::age(francesca, 27).
0.251::age(gina, 81).

0.211::it_age(some_male, 30).
0.223::it_age(colin, 41).
0.227::it_age(emilio, 27).
0.239::it_age(some_female, 23).
0.241::it_age(francesca, 27).
0.251::it_age(gina, 81).

0.307::height(some_male, 1.73).
0.311::height(colin, 1.57).
0.313::height(emilio, 2.06).
0.317::height(james, 1.70).
0.331::height(marco, 1.65).
0.337::height(some_female, 1.72).
0.347::height(francesca, 1.70).
0.349::height(gina, 1.45).
0.353::height(jennifer, 1.73).
0.359::height(lucia, 1.59).
0.367::height(pierro, 1.82).
0.373::height(charlotte, 1.81).

0.307::it_height(some_male, 1.73).
0.311::it_height(colin, 1.57).
0.313::it_height(emilio, 2.06).
0.317::it_height(james, 1.70).
0.331::it_height(marco, 1.65).
0.337::it_height(some_female, 1.72).
0.347::it_height(francesca, 1.70).
0.349::it_height(gina, 1.45).
0.353::it_height(jennifer, 1.73).
0.359::it_height(lucia, 1.59).
0.367::it_height(pierro, 1.82).
0.373::it_height(charlotte, 1.81).

0.211::inv_age(30, some_male).
0.223::inv_age(41, colin).
0.227::inv_age(27, emilio).
0.239::inv_age(23, some_female).
0.241::inv_age(27, francesca).
0.251::inv_age(81, gina).

0.211::it_inv_age(30, some_male).
0.223::it_inv_age(41, colin).
0.227::it_inv_age(27, emilio).
0.239::it_inv_age(23, some_female).
0.241::it_inv_age(27, francesca).
0.251::it_inv_age(81, gina).

0.307::inv_height(1.73, some_male).
0.311::inv_height(1.57, colin).
0.313::inv_height(2.06, emilio).
0.317::inv_height(1.70, james).
0.331::inv_height(1.65, marco).
0.337::inv_height(1.72, some_female).
0.347::inv_height(1.70, francesca).
0.349::inv_height(1.45, gina).
0.353::inv_height(1.73, jennifer).
0.359::inv_height(1.65, lucia).
0.367::inv_height(1.65, pierro).
0.373::inv_height(1.81, charlotte).

0.307::it_inv_height(1.73, some_male).
0.311::it_inv_height(1.57, colin).
0.313::it_inv_height(2.06, emilio).
0.317::it_inv_height(1.70, james).
0.331::it_inv_height(1.65, marco).
0.337::it_inv_height(1.72, some_female).
0.347::it_inv_height(1.70, francesca).
0.349::it_inv_height(1.45, gina).
0.353::it_inv_height(1.73, jennifer).
0.359::it_inv_height(1.65, lucia).
0.367::it_inv_height(1.65, pierro).
0.373::it_inv_height(1.81, charlotte).

0.547::husband(andrew, christine).
0.557::husband(arthur, margaret).
0.563::husband(charles, jennifer).
0.569::husband(christopher, penelope).
0.571::husband(emilio, gina).
0.577::husband(james, victoria).
0.587::husband(marco, lucia).
0.593::husband(pierro, francesca).
0.599::husband(roberto, maria).
0.601::husband(tomaso, angela).
0.6012::husband(some_male, some_female).
0.7019::husband(james, some_female).

0.547::it_husband(andrew, christine).
0.557::it_husband(arthur, margaret).
0.563::it_husband(charles, jennifer).
0.569::it_husband(christopher, penelope).
0.571::it_husband(emilio, gina).
0.577::it_husband(james, victoria).
0.587::it_husband(marco, lucia).
0.593::it_husband(pierro, francesca).
0.599::it_husband(roberto, maria).
0.601::it_husband(tomaso, angela).
0.6012::it_husband(some_male, some_female).
0.7019::it_husband(james, some_female).

0.547::it_husband_2(andrew, christine).
0.557::it_husband_2(arthur, margaret).
0.563::it_husband_2(charles, jennifer).
0.569::it_husband_2(christopher, penelope).
0.571::it_husband_2(emilio, gina).
0.577::it_husband_2(james, victoria).
0.587::it_husband_2(marco, lucia).
0.593::it_husband_2(pierro, francesca).
0.599::it_husband_2(roberto, maria).
0.601::it_husband_2(tomaso, angela).
0.6012::it_husband_2(some_male, some_female).
0.7019::it_husband_2(james, some_female).

0.6015::wife(some_female, some_male).
0.607::wife(angela, tomaso).
0.613::wife(christine, andrew).
0.617::wife(francesca, pierro).
0.619::wife(gina, emilio).
0.631::wife(jennifer, charles).
0.641::wife(lucia, marco).
0.643::wife(maria, roberto).
0.647::wife(penelope, christopher).
0.653::wife(victoria, james).

0.701::father(andrew, james).
0.709::father(andrew, jennifer).
0.006::father(andrew, andrew).
0.719::father(christopher, arthur).
0.727::father(christopher, victoria).
0.733::father(james, charlotte).
0.739::father(james, colin).
0.743::father(marco, alfonso).
0.751::father(marco, sophia).
0.003::father(marco, marco).
0.757::father(pierro, angela).
0.761::father(pierro, marco).
0.769::father(roberto, emilio).
0.773::father(roberto, lucia).

0.787::mother(christine, jennifer).
0.797::mother(francesca, angela).
0.809::mother(francesca, marco).
0.811::mother(lucia, sophia).
0.821::mother(maria, emilio).
0.823::mother(maria, lucia).
0.827::mother(penelope, arthur).
0.829::mother(penelope, victoria).
0.839::mother(victoria, charlotte).
0.853::mother(victoria, colin).

0.859::son(alfonso, lucia).
0.863::son(alfonso, marco).
0.877::son(arthur, christopher).
0.881::son(arthur, penelope).
0.883::son(colin, james).
0.887::son(colin, victoria).
0.907::son(emilio, roberto).
0.911::son(james, andrew).
0.919::son(james, christine).
0.929::son(marco, francesca).
0.937::son(marco, pierro).

0.941::daughter(angela, pierro).
0.9415::daughter(some_female, jennifer).
0.947::daughter(charlotte, james).
0.094::daughter(jennifer, james).
0.953::daughter(charlotte, victoria).
0.967::daughter(jennifer, andrew).
0.971::daughter(jennifer, christine).
0.977::daughter(lucia, maria).
0.983::daughter(lucia, roberto).
0.991::daughter(sophia, lucia).
0.997::daughter(sophia, marco).
0.2::daughter(victoria, christopher).
0.3::daughter(victoria, penelope).

0.941::it_daughter(angela, pierro).
0.9415::it_daughter(some_female, jennifer).
0.947::it_daughter(charlotte, james).
0.094::it_daughter(jennifer, james).
0.953::it_daughter(charlotte, victoria).
0.967::it_daughter(jennifer, andrew).
0.971::it_daughter(jennifer, christine).
0.977::it_daughter(lucia, maria).
0.983::it_daughter(lucia, roberto).
0.991::it_daughter(sophia, lucia).
0.997::it_daughter(sophia, marco).
0.2::it_daughter(victoria, christopher).
0.3::it_daughter(victoria, penelope).

0.5::brother(alfonso, sophia).
0.7::brother(arthur, victoria).
0.11::brother(colin, charlotte).
0.13::brother(emilio, lucia).
0.17::brother(james, jennifer).
0.19::brother(marco, angela).
0.5853::brother(jennifer, some_female).
0.6853::brother(angela, some_female).
0.7853::brother(marco, some_female).

0.5::it_brother(alfonso, sophia).
0.7::it_brother(arthur, victoria).
0.11::it_brother(colin, charlotte).
0.13::it_brother(emilio, lucia).
0.17::it_brother(james, jennifer).
0.19::it_brother(marco, angela).
0.5853::it_brother(jennifer, some_female).
0.6853::it_brother(angela, some_female).
0.7853::it_brother(marco, some_female).

0.23::sister(angela, marco).
0.29::sister(charlotte, colin).
0.31::sister(lucia, emilio).
0.37::sister(sophia, alfonso).
0.41::sister(victoria, arthur).
0.5853::sister(some_female, jennifer).
0.6853::sister(some_female, angela).
0.7853::sister(some_female, marco).

0.23::it_sister(angela, marco).
0.29::it_sister(charlotte, colin).
0.31::it_sister(lucia, emilio).
0.37::it_sister(sophia, alfonso).
0.41::it_sister(victoria, arthur).
0.5853::it_sister(some_female, jennifer).
0.6853::it_sister(some_female, angela).
0.7853::it_sister(some_female, marco).

0.449::uncle(arthur, charlotte).
0.029::uncle(arthur, arthur).
0.049::uncle(some_male, some_female).
0.457::uncle(charles, charlotte).
0.461::uncle(charles, colin).
0.463::uncle(emilio, alfonso).
0.013::uncle(emilio, emilio).
0.467::uncle(emilio, sophia).
0.479::uncle(tomaso, alfonso).
0.487::uncle(tomaso, sophia).

0.449::it_uncle(arthur, charlotte).
0.029::it_uncle(arthur, arthur).
0.049::it_uncle(some_male, some_female).
0.457::it_uncle(charles, charlotte).
0.461::it_uncle(charles, colin).
0.463::it_uncle(emilio, alfonso).
0.013::it_uncle(emilio, emilio).
0.467::it_uncle(emilio, sophia).
0.479::it_uncle(tomaso, alfonso).
0.487::it_uncle(tomaso, sophia).

0.419::aunt(angela, alfonso).
0.421::aunt(gina, alfonso).
0.431::aunt(gina, sophia).
0.433::aunt(jennifer, charlotte).
0.0433::aunt(jennifer, some_male).
0.21::aunt(some_female, alfonso).
0.31::aunt(some_female, arthur).
0.33::aunt(some_female, charlotte).
0.439::aunt(jennifer, colin).
0.443::aunt(margaret, charlotte).
0.449::aunt(margaret, colin).

0.419::it_aunt(angela, alfonso).
0.421::it_aunt(gina, alfonso).
0.431::it_aunt(gina, sophia).
0.433::it_aunt(jennifer, charlotte).
0.0433::it_aunt(jennifer, some_male).
0.21::it_aunt(some_female, alfonso).
0.31::it_aunt(some_female, arthur).
0.33::it_aunt(some_female, charlotte).
0.439::it_aunt(jennifer, colin).
0.443::it_aunt(margaret, charlotte).
0.449::it_aunt(margaret, colin).

0.73::nephew(alfonso, angela).
0.079::nephew(alfonso, arthur).
0.083::nephew(alfonso, charlotte).
0.03::nephew(alfonso, alfonso).
0.79::nephew(alfonso, gina).
0.83::nephew(alfonso, tomaso).
0.89::nephew(colin, arthur).
0.012::nephew(colin, colin).
0.083::nephew(alfonso, tomaso).
0.089::nephew(colin, arthur).
0.97::nephew(colin, charles).
0.101::nephew(colin, jennifer).
0.103::nephew(colin, margaret).

0.107::niece(charlotte, arthur).
0.109::niece(charlotte, charles).
0.113::niece(charlotte, jennifer).
0.127::niece(charlotte, margaret).
0.0107::niece(jennifer, some_female).
0.0109::niece(lucia, some_female).
0.0149::niece(lucia, jennifer).
0.0113::niece(angela, some_female).
0.0131::niece(angela, jennifer).
0.0127::niece(charlotte, some_female).
0.131::niece(sophia, angela).
0.137::niece(sophia, emilio).
0.139::niece(sophia, gina).
0.149::niece(sophia, tomaso).

0.107::it_niece(charlotte, arthur).
0.109::it_niece(charlotte, charles).
0.113::it_niece(charlotte, jennifer).
0.127::it_niece(charlotte, margaret).
0.0107::it_niece(jennifer, some_female).
0.0109::it_niece(lucia, some_female).
0.0149::it_niece(lucia, jennifer).
0.0113::it_niece(angela, some_female).
0.0131::it_niece(angela, jennifer).
0.0127::it_niece(charlotte, some_female).
0.131::it_niece(sophia, angela).
0.137::it_niece(sophia, emilio).
0.139::it_niece(sophia, gina).
0.149::it_niece(sophia, tomaso).

male_human(X) :- it_male(X).
female_human(X) :- it_female(X).

has_inv_height(X, Y) :- it_inv_height(X, Y).
has_height(X, Y) :- it_height(X, Y).
has_age(X, Y) :- it_age(X, Y).
has_inv_age(X, Y) :- it_inv_age(X, Y).

has_daughter(maria, Y) :- it_daughter(maria, Y).
has_sister(angela, Y) :- it_sister(angela, Y).
has_husband(X, victoria) :- it_husband(X, victoria).
has_husband_2(X, Y) :- it_husband_2(X, Y).
has_brother(X, angela) :- it_brother(X, angela).
has_aunt(X, Y) :- it_aunt(X, Y).
has_uncle(arthur, Y) :- it_uncle(arthur, Y).
has_niece(X, gina) :- it_niece(X, gina).
has_nephew(X, Y) :- nephew(X, X).
#has_nephew(X, Y) :- nephew(X, Y).

#human(X) :- {P}(X, Y), score(b_{P}).
#human(X) :- {P}(Y, X), score(b_{P}).
#
for x in father mother do
    for y in father mother do
        grand_{x}(X, Y) :- {x}(X, Z), {y}(Z, Y), w(grand_{x}).
    done
done
#
#grand_father(X, Y) :- father(X, Z), father(Z, Y), w(x, y).
