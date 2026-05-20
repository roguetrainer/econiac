<!-- markdownlint-disable MD033 -->
# Examples

Worked examples spanning macroeconomics, climate, supply chain, and financial gauge theory.
Each example links to a runnable notebook and a standalone Python script. Filter by domain or tag.

<div id="ec-filter-domain" style="margin: 1.5rem 0 0.75rem;">
  <span style="font-size: 0.85rem; font-weight: 600; margin-right: 0.5rem;">Domain:</span>
  <button class="ec-btn active" data-filter="domain" data-value="all">All</button>
  <button class="ec-btn" data-filter="domain" data-value="macroeconomics">Macroeconomics</button>
  <button class="ec-btn" data-filter="domain" data-value="climate">Climate</button>
  <button class="ec-btn" data-filter="domain" data-value="finance">Finance</button>
  <button class="ec-btn" data-filter="domain" data-value="supply-chain">Supply Chain</button>
  <button class="ec-btn" data-filter="domain" data-value="risk">Risk</button>
</div>

<div id="ec-filter-tags" style="margin: 0 0 1.5rem;">
  <span style="font-size: 0.85rem; font-weight: 600; margin-right: 0.5rem;">Tag:</span>
  <button class="ec-btn active" data-filter="tag" data-value="all">All</button>
  <button class="ec-btn" data-filter="tag" data-value="differentiable">Differentiable</button>
  <button class="ec-btn" data-filter="tag" data-value="sfc">SFC</button>
  <button class="ec-btn" data-filter="tag" data-value="climate">Climate</button>
  <button class="ec-btn" data-filter="tag" data-value="jax">JAX</button>
  <button class="ec-btn" data-filter="tag" data-value="gauge-theory">Gauge Theory</button>
  <button class="ec-btn" data-filter="tag" data-value="tir">TIR</button>
  <button class="ec-btn" data-filter="tag" data-value="shapley">Shapley</button>
  <button class="ec-btn" data-filter="tag" data-value="tropical">Tropical</button>
  <button class="ec-btn" data-filter="tag" data-value="pcl">PCL</button>
  <button class="ec-btn" data-filter="tag" data-value="carbon-tax">Carbon Tax</button>
</div>

<p id="ec-count" style="font-size: 0.8rem; color: #888; margin-bottom: 1rem;"></p>

<div id="ec-grid"></div>

<style>
.ec-btn {
  display: inline-block;
  margin: 0.15rem 0.2rem;
  padding: 0.2rem 0.65rem;
  font-size: 0.8rem;
  border: 1px solid #aaa;
  border-radius: 3px;
  background: var(--md-default-bg-color, #f5f5f5);
  color: var(--md-default-fg-color, #333);
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s;
}
.ec-btn:hover { background: var(--md-accent-fg-color--transparent, #e0e0e0); }
.ec-btn.active { background: #3f51b5; border-color: #3f51b5; color: #fff; }

#ec-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}

.ec-card {
  border: 1px solid var(--md-default-fg-color--lightest, #ddd);
  border-radius: 8px;
  padding: 1rem 1.1rem;
  background: var(--md-default-bg-color, #fff);
  transition: box-shadow 0.15s;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}
.ec-card:hover { box-shadow: 0 2px 10px rgba(0,0,0,0.10); }

.ec-card-title {
  font-weight: 600;
  font-size: 0.95rem;
  color: #3f51b5;
  text-decoration: none;
}
.ec-card-title:hover { text-decoration: underline; }

.ec-card-desc {
  font-size: 0.82rem;
  color: var(--md-default-fg-color--light, #555);
  line-height: 1.45;
}

.ec-card-links {
  font-size: 0.78rem;
  margin-top: 0.2rem;
}
.ec-card-links a {
  color: #3f51b5;
  text-decoration: none;
  margin-right: 0.75rem;
}
.ec-card-links a:hover { text-decoration: underline; }

.ec-tags { display: flex; flex-wrap: wrap; gap: 0.25rem; margin-top: 0.3rem; }

.ec-tag {
  font-size: 0.7rem;
  background: #eef2ff;
  border: 1px solid #c7d2fe;
  border-radius: 3px;
  padding: 0.05rem 0.35rem;
  color: #3730a3;
}

.ec-domain {
  font-size: 0.7rem;
  background: #f0fdf4;
  border: 1px solid #bbf7d0;
  border-radius: 3px;
  padding: 0.05rem 0.35rem;
  color: #166534;
}

.ec-difficulty {
  font-size: 0.68rem;
  border-radius: 3px;
  padding: 0.05rem 0.35rem;
  font-weight: 600;
}
.ec-difficulty-beginner { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
.ec-difficulty-intermediate { background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }
.ec-difficulty-advanced { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
</style>

<script>
(function() {
  var examples = [
    {
      "slug": "moniac",
      "title": "MONIAC: The Hydraulic Economy, Differentiable",
      "description": "Replicates Phillips\u2019s 1949 hydraulic computer in EconIAC. PCL conservation, Gibbs slot-cam, jax.grad fiscal multiplier, accelerator bifurcation \u03c7(\u03b2) early-warning.",
      "notebook": "tutorials/moniac.ipynb",
      "domain": ["macroeconomics"],
      "tags": ["differentiable", "sfc", "gibbs", "jax", "tir", "bifurcation"],
      "difficulty": "beginner"
    },
    {
      "slug": "keen-predator-prey",
      "title": "Keen Predator-Prey",
      "description": "Debt-deflation dynamics as a predator-prey ODE. Introduces BalanceSheet, PCL choose, TIR routing, and thermal Shapley attribution.",
      "notebook": "tutorials/keen_predator_prey.ipynb",
      "domain": ["macroeconomics"],
      "tags": ["ode", "debt", "sfc", "tir", "shapley"],
      "difficulty": "beginner"
    },
    {
      "slug": "gemmes",
      "title": "GEMMES: Keen + Climate",
      "description": "Keen macro model extended with climate damage curvature, stranded assets, and a 4-player carbon-tax Shapley attribution.",
      "notebook": "tutorials/gemmes.ipynb",
      "domain": ["macroeconomics", "climate"],
      "tags": ["climate", "ode", "sfc", "shapley", "carbon-tax"],
      "difficulty": "intermediate"
    },
    {
      "slug": "gl-pc",
      "title": "GL Model PC: Portfolio Choice",
      "description": "Godley-Lavoie portfolio-choice model with Gibbs-weighted asset allocation. Calibrates \u03b2* from Flow of Funds data.",
      "notebook": "tutorials/gl_pc.ipynb",
      "domain": ["macroeconomics", "finance"],
      "tags": ["sfc", "portfolio", "gibbs", "calibration"],
      "difficulty": "intermediate"
    },
    {
      "slug": "lowgrow",
      "title": "LowGrow SSE: Green Transition",
      "description": "Low-growth steady-state economy with TIR investment routing, carbon lock-in paradox, and green-transition phase diagram.",
      "notebook": "tutorials/lowgrow.ipynb",
      "domain": ["macroeconomics", "climate"],
      "tags": ["climate", "sfc", "tir", "carbon-tax", "phase-diagram"],
      "difficulty": "intermediate"
    },
    {
      "slug": "supply-chain-rst",
      "title": "Supply Chain RST",
      "description": "Differentiable reverse stress test on a copper supply chain. SupplyCapacity (AND/product) vs FinancialRisk (OR/sum) with Curry-Howard duality.",
      "notebook": "tutorials/supply_chain.ipynb",
      "domain": ["supply-chain", "risk"],
      "tags": ["differentiable", "rst", "jax", "tropical", "pcl"],
      "difficulty": "advanced"
    },
    {
      "slug": "climate-yield",
      "title": "Climate Hazard Yield Surface",
      "description": "Computes the 2-D investment yield surface \u03a6(t_inv, t_pay) and the doomsday clock isocurve, translated to per-household dollar cost.",
      "notebook": null,
      "domain": ["climate", "risk"],
      "tags": ["climate", "yield-surface", "geometry", "policy"],
      "difficulty": "intermediate"
    },
    {
      "slug": "triangular-arbitrage",
      "title": "Triangular Arbitrage",
      "description": "FX triangular arbitrage as holonomy on the Pacioli manifold. Connection curvature detects arbitrage opportunities in currency triples.",
      "notebook": null,
      "domain": ["finance"],
      "tags": ["fx", "gauge-theory", "holonomy", "curvature"],
      "difficulty": "advanced"
    }
  ];

  var activeDomain = "all";
  var activeTag = "all";

  function render() {
    var visible = examples.filter(function(e) {
      var domainOk = activeDomain === "all" || e.domain.indexOf(activeDomain) !== -1;
      var tagOk = activeTag === "all" || e.tags.indexOf(activeTag) !== -1;
      return domainOk && tagOk;
    });

    var grid = document.getElementById("ec-grid");
    var count = document.getElementById("ec-count");
    count.textContent = visible.length + " example" + (visible.length !== 1 ? "s" : "");

    grid.innerHTML = visible.map(function(e) {
      var domains = e.domain.map(function(d) {
        return '<span class="ec-domain">' + d + '</span>';
      }).join("");
      var tags = e.tags.map(function(t) {
        return '<span class="ec-tag">' + t + '</span>';
      }).join("");
      var diffClass = "ec-difficulty ec-difficulty-" + e.difficulty;
      var links = "";
      if (e.notebook) {
        links += '<a href="' + e.notebook + '">Notebook</a>';
      }
      links += '<a href="https://github.com/roguetrainer/econiac/blob/main/examples/' + e.slug.replace(/-/g, "_") + '.py" target="_blank">Script</a>';

      return '<div class="ec-card">'
        + '<div style="display:flex; justify-content:space-between; align-items:flex-start;">'
        + '<span class="ec-card-title">' + e.title + '</span>'
        + '<span class="' + diffClass + '">' + e.difficulty + '</span>'
        + '</div>'
        + '<div class="ec-card-desc">' + e.description + '</div>'
        + '<div class="ec-tags">' + domains + tags + '</div>'
        + '<div class="ec-card-links">' + links + '</div>'
        + '</div>';
    }).join("");

    if (visible.length === 0) {
      grid.innerHTML = '<p style="color:#888; font-size:0.9rem;">No examples match the current filters.</p>';
    }
  }

  function bindButtons(containerId, filterKey) {
    var container = document.getElementById(containerId);
    if (!container) return;
    container.addEventListener("click", function(ev) {
      var btn = ev.target.closest(".ec-btn");
      if (!btn || btn.dataset.filter !== filterKey) return;
      container.querySelectorAll(".ec-btn").forEach(function(b) { b.classList.remove("active"); });
      btn.classList.add("active");
      if (filterKey === "domain") activeDomain = btn.dataset.value;
      else activeTag = btn.dataset.value;
      render();
    });
  }

  bindButtons("ec-filter-domain", "domain");
  bindButtons("ec-filter-tags", "tag");
  render();
})();
</script>
