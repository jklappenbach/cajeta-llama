/*
 * ChatTemplate.g4 — the normative grammar of the chat-template dialect
 * (cajeta-llama spec 7.11–7.14, 13.18).
 *
 * STATUS: this grammar is the CONTRACT, not a code-generator input.
 * ANTLR4 ships no cajeta target and there is no ANTLR runtime on the
 * cajeta side (every parser written in cajeta is hand-written recursive
 * descent: stdlib `JsonReader`, `ProtobufCursor`, `dev.cajeta.docs`'s
 * Html/Markdown readers). `TplParser.cajeta` implements this grammar by
 * hand; drift between the two is bounded by the byte-exact parity tests
 * against `transformers.apply_chat_template` (spec 7.14). If a cajeta
 * ANTLR target ever lands, this file is the input it consumes.
 *
 * SCOPE is the chat-template dialect, not Jinja: what the four target
 * families and the tool-call templates actually use. A construct outside
 * this grammar is an ERROR naming the construct and the line (7.12) —
 * never a silent partial render.
 *
 * WHITESPACE semantics match `transformers`' environment exactly:
 * ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).
 *   - trim_blocks:   a newline IMMEDIATELY after a block or comment tag
 *                    is removed.
 *   - lstrip_blocks: horizontal whitespace from line start up to a block
 *                    or comment tag is removed.
 *   - Neither applies to `{{ }}` output tags.
 *   - Explicit `{%-`/`-%}`/`{{-`/`-}}`/`{#-`/`-#}` strip ALL adjacent
 *     whitespace including newlines and override the above.
 */

grammar ChatTemplate;

// ── document ────────────────────────────────────────────────────────────

template    : node* EOF ;

node        : text
            | output
            | ifStmt
            | forStmt
            | setStmt
            | macroStmt
            | comment
            ;

text        : TEXT ;
comment     : COMMENT_OPEN .*? COMMENT_CLOSE ;
output      : VAR_OPEN expr VAR_CLOSE ;

// ── statements ──────────────────────────────────────────────────────────

ifStmt      : blockOpen 'if' expr blockClose
              node*
              elifClause*
              elseClause?
              blockOpen 'endif' blockClose
            ;
elifClause  : blockOpen 'elif' expr blockClose node* ;
elseClause  : blockOpen 'else' blockClose node* ;

forStmt     : blockOpen 'for' nameList 'in' expr blockClose
              node*
              elseClause?                    // for/else: empty-sequence arm
              blockOpen 'endfor' blockClose
            ;
nameList    : NAME (',' NAME)* ;

setStmt     : blockOpen 'set' NAME '=' expr blockClose ;

macroStmt   : blockOpen 'macro' NAME '(' paramList? ')' blockClose
              node*
              blockOpen 'endmacro' blockClose
            ;
paramList   : param (',' param)* ;
param       : NAME ('=' expr)? ;             // defaulted parameters

blockOpen   : STMT_OPEN | STMT_OPEN_TRIM ;   // '{%'  | '{%-'
blockClose  : STMT_CLOSE | STMT_CLOSE_TRIM ; // '%}'  | '-%}'

// ── expressions (lowest to highest precedence) ──────────────────────────

expr        : condExpr ;
condExpr    : orExpr ('if' orExpr ('else' condExpr)?)? ;   // inline if
orExpr      : andExpr ('or' andExpr)* ;
andExpr     : notExpr ('and' notExpr)* ;
notExpr     : 'not' notExpr | cmpExpr ;
cmpExpr     : concatExpr (cmpOp concatExpr | 'is' 'not'? NAME testArgs?)* ;
cmpOp       : '==' | '!=' | '<' | '>' | '<=' | '>=' | 'in' | 'not' 'in' ;
concatExpr  : addExpr ('~' addExpr)* ;                     // string concat
addExpr     : mulExpr (('+' | '-') mulExpr)* ;
mulExpr     : unary (('*' | '/' | '//' | '%') unary)* ;
unary       : '-' unary | postfix ;
postfix     : primary trailer* ;
trailer     : '.' NAME                                     // attribute
            | '[' subscript ']'                            // index / slice
            | '(' argList? ')'                             // call
            | '|' NAME ('(' argList? ')')?                 // filter
            ;
subscript   : expr | slice ;
slice       : expr? ':' expr? ;
argList     : arg (',' arg)* ;
arg         : (NAME '=')? expr ;                           // keyword args
testArgs    : '(' argList? ')' ;

primary     : NUMBER
            | STRING
            | 'true' | 'false' | 'none'
            | NAME
            | listLit
            | dictLit
            | '(' expr ')'
            ;
listLit     : '[' (expr (',' expr)*)? ']' ;
dictLit     : '{' (pair (',' pair)*)? '}' ;
pair        : expr ':' expr ;

/*
 * DEFINED NAMES resolved by the interpreter (not grammar):
 *   variables      messages, tools, add_generation_prompt, bos_token,
 *                  eos_token, plus any `set` name and macro parameters;
 *                  `loop` inside a for body (index0, index, first, last,
 *                  revindex0, length).
 *   filters        tojson, trim, join, default, length, lower, upper,
 *                  replace, string, int, list, safe (identity — no
 *                  autoescape in this dialect), selectattr-free.
 *   tests          defined, none, string, mapping, iterable, sequence,
 *                  number, boolean, true, false.
 *   methods        .items(), .get(k[,default]), .keys(), .values(),
 *                  .append(v), .split([sep]), .strip(), .startswith(s),
 *                  .endswith(s), .lstrip(), .rstrip(), .rsplit, .title()
 *   functions      raise_exception(msg)  -> engine error with msg (7.12)
 *                  strftime_now(fmt)     -> injectable clock (13.1.8)
 *                  namespace(**kw)       -> mutable holder object
 *                  range(a[,b[,c]])
 */

// ── lexer ───────────────────────────────────────────────────────────────

VAR_OPEN        : '{{' '-'? ;
VAR_CLOSE       : '-'? '}}' ;
STMT_OPEN       : '{%' ;
STMT_OPEN_TRIM  : '{%-' ;
STMT_CLOSE      : '%}' ;
STMT_CLOSE_TRIM : '-%}' ;
COMMENT_OPEN    : '{#' '-'? ;
COMMENT_CLOSE   : '-'? '#}' ;

NAME    : [a-zA-Z_] [a-zA-Z_0-9]* ;
NUMBER  : [0-9]+ ('.' [0-9]+)? ;
STRING  : '\'' (~['\\] | '\\' .)* '\''
        | '"'  (~["\\] | '\\' .)* '"'
        ;
TEXT    : ~[{]+ | '{' ~[{%#] ;      // any run not opening a tag
WS      : [ \t\r\n]+ -> skip ;      // inside tags only; TEXT keeps its own
