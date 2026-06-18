# Chapter 2: The Balance Sheet as a Simplicial Complex

> *A node is a firm. An edge is a contract. A triangle is a cycle of obligations.*

---

## The example

Three banks: A, B, C. A has lent £100 to B. B has lent £80 to C. C has posted £60 of collateral back to A.

Draw this as a graph: three nodes, three directed edges. This is the **obligation complex** $\Gamma$ for this three-firm system.

Each edge is a bilateral contract — a promise from one party to another. Taken individually, each is well-defined and enforceable. But the three together form a triangle, and triangles have properties that edges do not.

---

## Simplices

A **simplicial complex** is built from:

- **0-simplices** (nodes): financial institutions, accounts, currencies
- **1-simplices** (edges): bilateral contracts, loans, swaps, repos
- **2-simplices** (triangles): three mutually connected nodes — a funding cycle
- **3-simplices** (tetrahedra): four mutually connected nodes — a four-party cycle

The balance sheet of the financial system is the 0- and 1-skeleton of this complex: nodes and edges only. Risk management asks whether higher simplices — triangles and tetrahedra — are present, and if so, whether they are *filled* or *hollow*.

---

## Filled vs hollow

A **filled triangle** {A, B, C} means there exists a direct three-way settlement mechanism: if A, B, and C all default simultaneously, a single clearing agent can resolve all three claims at once. CCP clearing partially achieves this for standardised products.

A **hollow triangle** {A, B, C} means the three bilateral contracts exist but no three-way settlement mechanism does. The triangle is present as a cycle in the graph, but there is no 2-simplex filling it in. The obligations are mutually consistent (each bilateral contract is valid) but there is no agent who can see and resolve all three simultaneously.

This distinction — filled vs hollow — is the geometric content of the difference between bilateral risk ($H^0$) and triangular risk ($H^1$).

---

## The Betti numbers

Given the obligation complex $\Gamma$, three numbers summarise its topology:

| Number | Name | Financial meaning |
|---|---|---|
| $\beta_0$ | Connected components | Number of isolated sub-networks |
| $\beta_1$ | Independent cycles | Number of unfilled funding loops |
| $\beta_2$ | Hollow voids | Number of irresolvable four-party conflicts |

$\beta_0 = 1$ means the network is connected — every firm can reach every other through some chain of obligations. $\beta_1 = 0$ means every funding loop is filled — no unresolved cycles. $\beta_2 = 0$ means every four-party conflict can be resolved — no hollow tetrahedra.

The 2008 financial crisis had $\beta_2 > 0$. The rest of this book explains what that means and why it matters.

---

## What standard risk management sees

A standard risk model operates on edges only: it measures the exposure on each bilateral contract, stress-tests each counterparty, and aggregates. It does not ask whether the triangle formed by three bilateral contracts is filled or hollow. It cannot see $\beta_1$ or $\beta_2$.

This is not a limitation of computational power. It is a conceptual limitation: the standard framework has no language for the topology of the network. Providing that language is the purpose of this book.

---

*Next: [Chapter 3 — When Do Bilateral Rates Fit Together?](ch03_h0_global_sections.md)*
