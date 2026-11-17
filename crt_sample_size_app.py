"""
Cluster Randomized Trial (CRT) Sample Size Calculator
Continuous and Binary (proportion) outcomes, parallel two-arm design.

Run:   pip install streamlit scipy pandas numpy
       streamlit run crt_sample_size_app.py
"""

import math

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm, t as t_dist

st.set_page_config(page_title="CRT Sample Size Calculator", page_icon="📊", layout="wide")


# ----------------------------------------------------------------------------
# Core statistics
# ----------------------------------------------------------------------------
def crit_values(alpha, power, two_sided, df=None):
    """Critical values for alpha and power (normal, or t if df given)."""
    a = alpha / 2 if two_sided else alpha
    if df is None:
        return norm.ppf(1 - a), norm.ppf(power)
    return t_dist.ppf(1 - a, df), t_dist.ppf(power, df)


def n_individual_control(outcome, p, r, za, zb):
    """
    Individually randomised sample size in the CONTROL arm.
    r = n_intervention / n_control (allocation ratio).
    """
    if outcome == "Continuous":
        delta = abs(p["mu1"] - p["mu2"])
        sd = p["sd"]
        return (1 + 1 / r) * sd**2 * (za + zb) ** 2 / delta**2
    p1, p2 = p["p1"], p["p2"]
    delta = abs(p1 - p2)
    pbar = (p1 + r * p2) / (1 + r)
    num = (
        za * math.sqrt((1 + 1 / r) * pbar * (1 - pbar))
        + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / r)
    ) ** 2
    return num / delta**2


def design_effect(m, rho, cv=0.0):
    """DEFF = 1 + ((CV^2 + 1) m - 1) * rho"""
    return 1 + ((cv**2 + 1) * m - 1) * rho


def solve_clusters(outcome, p, r, alpha, power, two_sided, rho, m, cv, use_t):
    """Number of clusters per arm for a given (average) cluster size m."""
    za, zb = crit_values(alpha, power, two_sided)
    n1 = n_individual_control(outcome, p, r, za, zb)
    deff = design_effect(m, rho, cv)
    k1 = max(2, math.ceil(n1 * deff / m))
    if use_t:
        for _ in range(100):
            k2 = math.ceil(r * k1)
            df = max(1, k1 + k2 - 2)
            za_t, zb_t = crit_values(alpha, power, two_sided, df)
            n1 = n_individual_control(outcome, p, r, za_t, zb_t)
            k_new = max(2, math.ceil(n1 * deff / m))
            if k_new <= k1:
                break
            k1 = k_new
    k2 = max(2, math.ceil(r * k1))
    return dict(n_ind_control=n1, n_ind_int=r * n1, deff=deff, k1=k1, k2=k2, za=za, zb=zb)


def solve_cluster_size(outcome, p, r, alpha, power, two_sided, rho, k1, cv, use_t):
    """Required average cluster size m for fixed clusters in control arm k1."""
    k2 = max(2, math.ceil(r * k1))
    if use_t:
        df = max(1, k1 + k2 - 2)
        za, zb = crit_values(alpha, power, two_sided, df)
    else:
        za, zb = crit_values(alpha, power, two_sided)
    n1 = n_individual_control(outcome, p, r, za, zb)
    denom = k1 - n1 * rho * (cv**2 + 1)
    if denom <= 0:
        return dict(feasible=False, n_ind_control=n1, min_k=n1 * rho * (cv**2 + 1), za=za, zb=zb)
    m = n1 * (1 - rho) / denom
    return dict(
        feasible=True, m=m, n_ind_control=n1, n_ind_int=r * n1,
        deff=design_effect(math.ceil(m), rho, cv), k1=k1, k2=k2,
        min_k=n1 * rho * (cv**2 + 1), za=za, zb=zb,
    )


def power_given_design(outcome, p, alpha, two_sided, rho, k1, k2, m, cv, use_t):
    """Power for fixed k1, k2, m."""
    deff = design_effect(m, rho, cv)
    n1 = k1 * m / deff
    n2 = k2 * m / deff
    r = n2 / n1
    if use_t:
        df = max(1, k1 + k2 - 2)
        za, _ = crit_values(alpha, 0.5, two_sided, df)
        cdf = lambda x: t_dist.cdf(x, df)
    else:
        za, _ = crit_values(alpha, 0.5, two_sided)
        cdf = norm.cdf
    if outcome == "Continuous":
        delta = abs(p["mu1"] - p["mu2"])
        se = p["sd"] * math.sqrt(1 / n1 + 1 / n2)
        zb = delta / se - za
    else:
        p1, p2 = p["p1"], p["p2"]
        delta = abs(p1 - p2)
        pbar = (p1 + r * p2) / (1 + r)
        zb = (delta * math.sqrt(n1) - za * math.sqrt((1 + 1 / r) * pbar * (1 - pbar))) / math.sqrt(
            p1 * (1 - p1) + p2 * (1 - p2) / r
        )
    return cdf(zb), n1, n2, deff


# ----------------------------------------------------------------------------
# Sidebar: inputs
# ----------------------------------------------------------------------------
st.title("📊 Cluster Randomized Trial: Sample Size Calculator")
st.caption("Two-arm parallel CRT · continuous or binary (proportion) outcome · clusters, cluster size or power")

with st.sidebar:
    st.header("1. Outcome")
    outcome = st.radio("Outcome type", ["Continuous", "Binary (proportion)"], index=1)
    outcome = "Continuous" if outcome == "Continuous" else "Binary"

    params = {}
    if outcome == "Continuous":
        params["mu1"] = st.number_input("Mean, control arm", value=10.0, step=0.5, format="%.3f")
        params["mu2"] = st.number_input("Mean, intervention arm", value=8.0, step=0.5, format="%.3f")
        params["sd"] = st.number_input("Standard deviation (total, common)", value=5.0, min_value=0.001, step=0.5, format="%.3f")
    else:
        params["p1"] = st.number_input("Proportion, control arm (p₁)", value=0.40, min_value=0.001, max_value=0.999, step=0.01, format="%.3f")
        params["p2"] = st.number_input("Proportion, intervention arm (p₂)", value=0.20, min_value=0.001, max_value=0.999, step=0.01, format="%.3f")

    st.header("2. Error rates")
    alpha = st.number_input("Significance level (α)", value=0.05, min_value=0.001, max_value=0.20, step=0.01, format="%.3f")
    power = st.number_input("Power (1 − β)", value=0.80, min_value=0.50, max_value=0.999, step=0.05, format="%.3f")
    two_sided = st.checkbox("Two-sided test", value=True)

    st.header("3. Cluster parameters")
    rho = st.number_input("ICC (ρ)", value=0.05, min_value=0.0, max_value=0.99, step=0.01, format="%.4f")
    cv = st.number_input("CV of cluster sizes (0 = equal sizes)", value=0.0, min_value=0.0, max_value=2.0, step=0.1,
                         help="Coefficient of variation of cluster sizes. Typical values 0.3–0.7 when sizes vary.")

    st.header("4. Allocation and adjustments")
    ratio = st.number_input("Allocation ratio (intervention : control clusters)", value=1.0, min_value=0.25, max_value=4.0, step=0.25)
    dropout = st.number_input("Expected attrition (%)", value=0.0, min_value=0.0, max_value=60.0, step=5.0) / 100
    use_t = st.checkbox("Small-sample t-correction (recommended when clusters < 15/arm)", value=True)

# ----------------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------------
errors = []
if outcome == "Continuous" and params["mu1"] == params["mu2"]:
    errors.append("Means must differ.")
if outcome == "Binary" and params["p1"] == params["p2"]:
    errors.append("Proportions must differ.")
if errors:
    for e in errors:
        st.error(e)
    st.stop()

# Effect summary
if outcome == "Continuous":
    diff = params["mu1"] - params["mu2"]
    d = abs(diff) / params["sd"]
    st.info(f"**Effect:** mean difference = {diff:.3f}, standardised effect (Cohen's d) = {d:.3f}")
else:
    p1, p2 = params["p1"], params["p2"]
    rr = p2 / p1
    orr = (p2 / (1 - p2)) / (p1 / (1 - p1))
    st.info(f"**Effect:** risk difference = {p1 - p2:+.3f}, risk ratio = {rr:.3f}, odds ratio = {orr:.3f}")

# ----------------------------------------------------------------------------
# Mode
# ----------------------------------------------------------------------------
mode = st.radio(
    "What do you want to calculate?",
    ["Number of clusters (given cluster size)", "Cluster size (given number of clusters)", "Power (given clusters and size)"],
    horizontal=True,
)

tab_res, tab_steps, tab_sens, tab_report = st.tabs(["✅ Result", "🧮 Step-by-step", "📈 Sensitivity", "📝 Protocol text"])

result_rows = []   # for download
report_text = ""
steps = []

# ----------------------------------------------------------------------------
# Mode 1: clusters needed
# ----------------------------------------------------------------------------
if mode.startswith("Number"):
    m = st.number_input("Average cluster size (m, analysable participants per cluster)", value=50, min_value=2, step=5)
    res = solve_clusters(outcome, params, ratio, alpha, power, two_sided, rho, m, cv, use_t)
    k1, k2 = res["k1"], res["k2"]
    n_enrol_per_cluster = math.ceil(m / (1 - dropout))
    tot_ctrl = k1 * n_enrol_per_cluster
    tot_int = k2 * n_enrol_per_cluster

    with tab_res:
        c = st.columns(4)
        c[0].metric("Clusters, control", k1)
        c[1].metric("Clusters, intervention", k2)
        c[2].metric("Total clusters", k1 + k2)
        c[3].metric("Total participants enrolled", tot_ctrl + tot_int)
        c = st.columns(4)
        c[0].metric("Design effect", f"{res['deff']:.2f}")
        c[1].metric("Individual-randomisation n / arm", f"{math.ceil(res['n_ind_control'])}")
        c[2].metric("Enrolled per cluster", n_enrol_per_cluster)
        c[3].metric("Effective n / arm (control)", f"{k1 * m / res['deff']:.0f}")
        if k1 < 6:
            st.warning("Fewer than 6 clusters per arm: the analysis is fragile and one unusual cluster can dominate. "
                       "Consider smaller clusters and more of them, or a cluster-level / small-sample-corrected analysis.")
        table = pd.DataFrame({
            "Arm": ["Control", "Intervention", "Total"],
            "Clusters": [k1, k2, k1 + k2],
            "Participants (analysable)": [k1 * m, k2 * m, (k1 + k2) * m],
            f"Participants (enrolled, {dropout:.0%} attrition)": [tot_ctrl, tot_int, tot_ctrl + tot_int],
        })
        st.dataframe(table, hide_index=True, width='stretch')
        result_rows = table

    za, zb = res["za"], res["zb"]
    steps = [
        ("Step 1: Sample size if individuals were randomised (control arm)", None),
    ]
    with tab_steps:
        st.subheader("Step 1. Individually randomised sample size")
        st.write(f"Critical values: Z(α) = {za:.4f}, Z(β) = {zb:.4f}; allocation ratio r = {ratio}")
        if outcome == "Continuous":
            st.latex(r"n_{1}=\frac{(1+1/r)\,\sigma^{2}\,(Z_{\alpha}+Z_{\beta})^{2}}{\delta^{2}}")
            delta = abs(params["mu1"] - params["mu2"])
            st.write(f"= (1 + 1/{ratio}) × {params['sd']}² × ({za:.4f} + {zb:.4f})² / {delta:.3f}² = **{res['n_ind_control']:.2f}**")
        else:
            st.latex(r"n_{1}=\frac{\left[Z_{\alpha}\sqrt{(1+1/r)\bar p(1-\bar p)}+Z_{\beta}\sqrt{p_1(1-p_1)+p_2(1-p_2)/r}\right]^{2}}{(p_1-p_2)^{2}},\quad \bar p=\frac{p_1+r\,p_2}{1+r}")
            pbar = (params["p1"] + ratio * params["p2"]) / (1 + ratio)
            st.write(f"p̄ = {pbar:.4f}, n₁ = **{res['n_ind_control']:.2f}** per control arm, n₂ = r × n₁ = {res['n_ind_int']:.2f}")

        st.subheader("Step 2. Design effect")
        st.latex(r"DEFF = 1 + \left[(CV^{2}+1)\,m - 1\right]\rho")
        st.write(f"= 1 + [({cv}² + 1) × {m} − 1] × {rho} = **{res['deff']:.3f}**")

        st.subheader("Step 3. Inflate and convert to clusters")
        st.latex(r"k_{1}=\left\lceil \frac{n_{1}\times DEFF}{m} \right\rceil")
        st.write(f"= ⌈{res['n_ind_control']:.2f} × {res['deff']:.3f} / {m}⌉ = ⌈{res['n_ind_control'] * res['deff'] / m:.2f}⌉ "
                 f"→ **{k1}** clusters in control; intervention k₂ = ⌈r × k₁⌉ = **{k2}**")
        if use_t:
            st.write("With the t-correction, Z-values were replaced by t-values with df = k₁ + k₂ − 2 and the calculation was iterated until the number of clusters stopped increasing.")

        st.subheader("Step 4. Attrition")
        st.latex(r"m_{enrol}=\left\lceil \frac{m}{1-\text{attrition}}\right\rceil")
        st.write(f"= ⌈{m} / (1 − {dropout:.2f})⌉ = **{n_enrol_per_cluster}** per cluster")

    # Sensitivity
    with tab_sens:
        st.subheader("Clusters per arm (control) by ICC and cluster size")
        icc_list = sorted(set([0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, round(rho, 4)]))
        m_list = sorted(set([10, 20, 30, 50, 100, 200, 400, 600, int(m)]))
        grid = pd.DataFrame(
            {f"m={mm}": [solve_clusters(outcome, params, ratio, alpha, power, two_sided, ic, mm, cv, use_t)["k1"] for ic in icc_list]
             for mm in m_list},
            index=[f"ICC={ic}" for ic in icc_list],
        )
        st.dataframe(grid, width='stretch')

        st.subheader("Trade-off: cluster size vs clusters vs total participants")
        ms = list(range(5, 301, 5))
        rows = []
        for mm in ms:
            rr_ = solve_clusters(outcome, params, ratio, alpha, power, two_sided, rho, mm, cv, use_t)
            rows.append({"m": mm, "Clusters per arm (control)": rr_["k1"],
                         "Total participants": (rr_["k1"] + rr_["k2"]) * mm})
        tdf = pd.DataFrame(rows).set_index("m")
        c1, c2 = st.columns(2)
        c1.write("Clusters per arm (control)")
        c1.line_chart(tdf["Clusters per arm (control)"])
        c2.write("Total participants (analysable)")
        c2.line_chart(tdf["Total participants"])
        st.caption(f"Minimum clusters as m → ∞ is about n₁·ρ·(CV²+1) = {res['n_ind_control'] * rho * (cv**2 + 1):.1f} per arm: adding participants to each cluster cannot reduce clusters below this.")

    sided = "two-sided" if two_sided else "one-sided"
    if outcome == "Binary":
        eff_txt = f"a reduction in the primary outcome proportion from {params['p1']:.0%} to {params['p2']:.0%}" if params["p2"] < params["p1"] else f"an increase in the primary outcome proportion from {params['p1']:.0%} to {params['p2']:.0%}"
    else:
        eff_txt = f"a difference in means of {abs(params['mu1'] - params['mu2']):.2f} (SD {params['sd']:.2f})"
    report_text = (
        f"Sample size was calculated for a two-arm parallel cluster randomised trial to detect {eff_txt}, "
        f"using a {sided} test at the {alpha:.0%} significance level with {power:.0%} power. "
        f"Assuming an intracluster correlation coefficient of {rho}"
        f"{f' and a coefficient of variation of cluster size of {cv}' if cv > 0 else ''} and an average of {m} analysable participants per cluster, "
        f"the design effect is {res['deff']:.2f}. The individually randomised sample size of {math.ceil(res['n_ind_control'])} per arm "
        f"is therefore inflated to {math.ceil(res['n_ind_control'] * res['deff'])} per arm, requiring {k1} clusters in the control arm and {k2} in the intervention arm "
        f"({k1 + k2} clusters in total{', with a small-sample t-correction' if use_t else ''}). "
        f"Allowing for {dropout:.0%} attrition, {n_enrol_per_cluster} participants will be enrolled per cluster "
        f"({tot_ctrl + tot_int} in total)."
    )

# ----------------------------------------------------------------------------
# Mode 2: cluster size needed
# ----------------------------------------------------------------------------
elif mode.startswith("Cluster size"):
    k1 = st.number_input("Number of clusters in the control arm (k₁)", value=10, min_value=2, step=1)
    res = solve_cluster_size(outcome, params, ratio, alpha, power, two_sided, rho, int(k1), cv, use_t)
    with tab_res:
        if not res["feasible"]:
            st.error(
                f"Not feasible: with ρ = {rho}, at least **{res['min_k']:.1f}** clusters per (control) arm are needed "
                f"even with unlimited cluster size. Increase the number of clusters."
            )
        else:
            m_need = math.ceil(res["m"])
            n_enrol = math.ceil(m_need / (1 - dropout))
            k2 = res["k2"]
            c = st.columns(4)
            c[0].metric("Required cluster size (analysable)", m_need)
            c[1].metric("Enrol per cluster", n_enrol)
            c[2].metric("Total clusters", int(k1) + k2)
            c[3].metric("Total participants enrolled", (int(k1) + k2) * n_enrol)
            st.dataframe(pd.DataFrame({
                "Arm": ["Control", "Intervention", "Total"],
                "Clusters": [int(k1), k2, int(k1) + k2],
                "Participants (enrolled)": [int(k1) * n_enrol, k2 * n_enrol, (int(k1) + k2) * n_enrol],
            }), hide_index=True, width='stretch')
            if m_need > 500:
                st.warning("Very large clusters are needed: consider adding clusters instead; returns diminish rapidly with ICC > 0.")
            result_rows = None
    with tab_steps:
        st.subheader("Formula")
        st.latex(r"m=\frac{n_{1}(1-\rho)}{k_{1}-n_{1}\,\rho\,(CV^{2}+1)}")
        st.write(f"n₁ (individually randomised, control arm) = {res['n_ind_control']:.2f}; "
                 f"minimum feasible clusters per arm = n₁·ρ·(CV²+1) = {res['min_k']:.2f}")
        if res["feasible"]:
            st.write(f"m = {res['n_ind_control']:.2f} × (1 − {rho}) / ({int(k1)} − {res['min_k']:.2f}) = **{res['m']:.2f}** → {math.ceil(res['m'])}")
    with tab_sens:
        st.subheader("Required cluster size by number of control clusters and ICC")
        icc_list = sorted(set([0.01, 0.02, 0.05, 0.10, round(rho, 4)]))
        k_list = sorted(set([4, 6, 8, 10, 15, 20, 30, 50, int(k1)]))
        out = {}
        for kk in k_list:
            col = []
            for ic in icc_list:
                rr_ = solve_cluster_size(outcome, params, ratio, alpha, power, two_sided, ic, kk, cv, use_t)
                col.append(math.ceil(rr_["m"]) if rr_["feasible"] else "n/a")
            out[f"k₁={kk}"] = col
        st.dataframe(pd.DataFrame(out, index=[f"ICC={i}" for i in icc_list]), width='stretch')
        st.caption("n/a = not achievable at any cluster size.")
    if res["feasible"]:
        report_text = (
            f"With {int(k1)} clusters in the control arm and {res['k2']} in the intervention arm, an average of "
            f"{math.ceil(res['m'])} analysable participants per cluster ({n_enrol} enrolled after {dropout:.0%} attrition) is required "
            f"to achieve {power:.0%} power at α = {alpha} ({'two' if two_sided else 'one'}-sided), assuming ICC = {rho}"
            f"{f' and CV of cluster size = {cv}' if cv > 0 else ''}."
        )

# ----------------------------------------------------------------------------
# Mode 3: power
# ----------------------------------------------------------------------------
else:
    c1, c2, c3 = st.columns(3)
    k1 = c1.number_input("Clusters, control (k₁)", value=10, min_value=2, step=1)
    k2 = c2.number_input("Clusters, intervention (k₂)", value=10, min_value=2, step=1)
    m = c3.number_input("Average analysable cluster size (m)", value=50, min_value=2, step=5)
    pw, n1, n2, deff = power_given_design(outcome, params, alpha, two_sided, rho, int(k1), int(k2), m, cv, use_t)
    with tab_res:
        c = st.columns(4)
        c[0].metric("Power", f"{pw:.1%}")
        c[1].metric("Design effect", f"{deff:.2f}")
        c[2].metric("Effective n, control", f"{n1:.0f}")
        c[3].metric("Effective n, intervention", f"{n2:.0f}")
        st.progress(min(max(pw, 0.0), 1.0))
        if pw < 0.8:
            st.warning("Power is below 80%.")
        else:
            st.success("Power is at least 80%.")
    with tab_steps:
        st.write(f"DEFF = 1 + [({cv}² + 1) × {m} − 1] × {rho} = {deff:.3f}")
        st.write(f"Effective sample size per arm = k × m / DEFF → control {n1:.1f}, intervention {n2:.1f}")
        if outcome == "Continuous":
            st.latex(r"Z_\beta=\frac{|\mu_1-\mu_2|}{\sigma\sqrt{1/n_1+1/n_2}}-Z_\alpha")
        else:
            st.latex(r"Z_\beta=\frac{|p_1-p_2|\sqrt{n_1}-Z_\alpha\sqrt{(1+1/r)\bar p(1-\bar p)}}{\sqrt{p_1(1-p_1)+p_2(1-p_2)/r}}")
        st.write("Power = Φ(Z_β) (or the t-distribution CDF if the small-sample correction is on).")
    with tab_sens:
        st.subheader("Power by clusters per arm and cluster size (equal arms)")
        k_list = sorted(set([4, 6, 8, 10, 15, 20, 30, int(k1)]))
        m_list = sorted(set([10, 20, 50, 100, 200, 600, int(m)]))
        g = pd.DataFrame(
            {f"m={mm}": [f"{power_given_design(outcome, params, alpha, two_sided, rho, kk, kk, mm, cv, use_t)[0]:.0%}" for kk in k_list]
             for mm in m_list},
            index=[f"k={kk}/arm" for kk in k_list],
        )
        st.dataframe(g, width='stretch')
    report_text = (
        f"With {int(k1)} control and {int(k2)} intervention clusters of {m} analysable participants on average, assuming ICC = {rho}"
        f"{f' and CV of cluster size = {cv}' if cv > 0 else ''} (design effect {deff:.2f}), the trial has {pw:.0%} power at α = {alpha} "
        f"({'two' if two_sided else 'one'}-sided) to detect the specified difference."
    )

# ----------------------------------------------------------------------------
# Protocol text + download
# ----------------------------------------------------------------------------
with tab_report:
    st.subheader("Sample size paragraph for your protocol")
    if report_text:
        st.text_area("Copy/paste", report_text, height=220)
    else:
        st.write("No feasible design with the current inputs.")
    inputs = {"outcome": outcome, **params, "alpha": alpha, "power": power, "two_sided": two_sided,
              "ICC": rho, "CV": cv, "allocation_ratio": ratio, "attrition": dropout, "t_correction": use_t, "mode": mode}
    st.download_button("⬇️ Download inputs (CSV)", pd.DataFrame([inputs]).to_csv(index=False), "crt_inputs.csv", "text/csv")

with st.expander("Assumptions and notes"):
    st.markdown(
        """
- **Design:** two-arm parallel CRT, analysis at the cluster level or with GEE/mixed models (equivalent under these approximations).
- **Continuous:** common SD across arms; the SD should be the *total* SD (between + within clusters).
- **Binary:** unpooled variance approximation. For multi-category outcomes, collapse to the primary binary contrast (or use an ordinal-outcome method).
- **ICC (ρ):** take from pilot data, previous trials in similar settings, or published ICC databases. Always run a sensitivity analysis.
- **Unequal cluster sizes:** DEFF = 1 + [(CV²+1)m − 1]ρ (Eldridge et al., 2006).
- **Small-sample t-correction:** replaces Z with t (df = k₁ + k₂ − 2) and iterates. Recommended when clusters < ~15 per arm.
- **Attrition:** participants are inflated per cluster; the number of clusters is not changed. Also allow for whole-cluster dropout if likely.
- Results should be verified in dedicated software (Stata `power`, R `clusterPower`/`CRTSize`, PASS, nQuery) before finalising a protocol.
        """
    )
