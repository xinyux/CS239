worksln(bill, sales).
worksln(sally, accounts).

deptManager(sales, joan).
deptManager(accounts, henry).

managerOf(joan, james).
managerOf(henry, james).
managerOf(james, paul).
managerOf(W, M) :- worksln(W, Dept), deptManager(Dept, M).

superiorOf(E, S) :- managerOf(E, S).
superiorOf(E, S) :- managerOf(E, M), superiorOf(M, S).

/** <query examples>

?- superiorOf(X, Y).
?- aggregate_all(count, (superiorOf(X, Y)), Count).
?- managerOf(X, Y).

*/