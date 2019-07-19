/**
 * Created on 12/07/2019.
 *
 * @author Victor Guimarães
 */

grammar NeuralLog;

//program: (clause)*;
program: (for_loop|clause)*;

for_loop: FOR_LOOP TERM IN_TOKEN
((for_term)+| '{' INTEGER '..' INTEGER '}')
DO_TOKEN
    (for_loop|clause)+
DONE_TOKEN;

clause: (atom | weighted_atom | horn_clause) END_OF_CLAUSE;

atom: (PLACE_HOLDER|TERM)+ list_of_arguments?;
weighted_atom: (number WEIGHT_SEPARATOR) atom;

horn_clause: atom IMPLICATION_SIGN (body)?;

body: literal (ITEM_SEPARATOR literal)*;

literal: NEGATION? TRAINABLE_IDENTIFIER? atom;

list_of_arguments:
    OPEN_ARGUMENTS argument (ITEM_SEPARATOR argument)* CLOSE_ARGUMENTS;

argument: number | term;

term: (QUOTED | (PLACE_HOLDER|TERM)+);

for_term: (term|number);

number: SCIENTIFIC_NUMBER | DECIMAL | INTEGER;


FOR_LOOP: 'for';
IN_TOKEN: 'in';
DO_TOKEN: 'do';
DONE_TOKEN: 'done';
NEGATION: 'not';
SCIENTIFIC_NUMBER: [0-9]* '.' [0-9]+ [eE][+-]? [0-9]+;
DECIMAL: [0-9]*'.'[0-9]+;
INTEGER: [0-9]+;
TERM: ([a-zA-Z0-9_-])+;
PLACE_HOLDER: '{' ([a-zA-Z0-9_-])+ '}';
OPEN_ARGUMENTS: '(';
CLOSE_ARGUMENTS: ')';
ITEM_SEPARATOR: ',';
END_OF_CLAUSE: '.';
WEIGHT_SEPARATOR: '::';
IMPLICATION_SIGN: ':-';
TRAINABLE_IDENTIFIER: '$';

QUOTED:
    	(
            '"'
            (
                '\\' .	//any escaped character
                |			//or
                ~["]		//any non-quote character
            )*
            '"'
            |
            '\''
            (
                '\\' .	//any escaped character
                |			//or
                ~[']		//any non-quote character
            )*
            '\''
        );

WHITESPACE: [ \t\r\n]+ -> skip ;
COMMENT:  ('#'|'%') ~( '\r' | '\n' )* -> skip;
BLOCK_COMMENT : '/*' .*? '*/' -> skip;
