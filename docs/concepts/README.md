# Concepts

How specific financial mechanisms work inside the EconIAC framework.

These pages sit between the abstract mathematical tools explained in
[Why does Econiac use abstract mathematics?](../why/README.md) and the
code-first [Tutorials](../tutorials/index.md). Each page takes one financial
mechanism and shows how the framework changes what can be said about it.

---

## Foundational concepts

These three ideas underpin everything else in EconIAC. Start here.

| Concept | The core claim | Read |
| --- | --- | --- |
| **Rationality is temperature** | Replacing argmax with softmax (β parameter) turns any economic model differentiable and calibratable. At β→∞ you recover the classical model exactly. | *(coming soon)* |
| **The three levels of risk** | Financial risk has three structural levels — bilateral (H⁰), triangular (H¹), systemic (H²) — and existing tools only address the first two. Options exist because H¹ ≠ 0. The 2008 crisis was H². | *(coming soon)* |
| **Clearing and netting** | Bilateral netting removes edges; CCP novation fills triangles; H² obstructions require a government-level instrument. The Eisenberg-Noe model is an H⁰ fixed-point computation — and what that means for what it can and cannot detect. | [Clearing, netting, and the topology of obligation](clearing.md) |

---

## Financial mechanisms

| Mechanism | What the framework adds | Read |
| --- | --- | --- |
| *(more coming soon)* | | |
