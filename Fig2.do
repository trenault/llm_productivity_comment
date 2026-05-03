clear all
set more off

* ─────────────────────────────────────────────────────────────────────────────
local N_AUTH    = 100000        // n_authors
local N_MON     = 30          // n_months
local T0        = 11          // intro_month
local BASE_MEAN = -1
local BASE_SD   = 1
local STATE_SD  = 0.5
local FLAG_PROB = 0.20
local N_REPS    = 1000         // Monte Carlo replications 
local W_PRE     = 12           
local W_POST    = 17     


* ─────────────────────────────────────────────────────────────────────────────
*  Postfile: collect one row per (scenario, replication, rel_time)
* ─────────────────────────────────────────────────────────────────────────────
capture postclose coef_collect
postfile coef_collect str8 scenario int rep_id int rel_time  ///
    double estimate se                                        ///
    using "sim_stata_coefs_100K_1000.dta", replace


* ─────────────────────────────────────────────────────────────────────────────
*  MAIN LOOP
* ─────────────────────────────────────────────────────────────────────────────
forvalues rep = 1 / `N_REPS' {

    di as result _n "── Rep `rep' / `N_REPS' ──"

    * ── (1) Generate panel ────────────────────────────────────────────────────
    clear
    set obs `= `N_AUTH' * `N_MON''

    gen int author_id = ceil(_n / `N_MON')
    gen int month     = mod(_n - 1, `N_MON') + 1

    * Author-level baseline: one draw per author, broadcast to all months
    sort author_id month
    by author_id: gen double mu_i = rnormal(`BASE_MEAN', `BASE_SD') if _n == 1
    by author_id: replace mu_i = mu_i[1]

    * i.i.d. monthly productivity count
    gen double log_lambda = rnormal(mu_i, `STATE_SD')
    gen long   count      = rpoisson(exp(log_lambda))

    * Detection: post-T0 only; prob = 1 - (1-FLAG_PROB)^count
    gen byte detected = 0
    replace  detected = (runiform() < 1 - (1 - `FLAG_PROB')^count) ///
        if month >= `T0' & count > 0

    * ── (2) Detection-scenario treatment timing ───────────────────────────────
    gen int _tmp = month if detected == 1
    bysort author_id: egen int first_det = min(_tmp)
    drop _tmp

    * ── (3) Random-scenario treatment timing (same N treated as detection) ────
    *  Count distinct treated authors in detection scenario
    bysort author_id (month): gen byte _fr = (_n == 1 & first_det < .)
    count if _fr
    local n_treated = r(N)
    drop _fr

    *  Draw random selection + timing
    preserve
        bysort author_id (month): keep if _n == 1
        keep author_id
        gen double _u = runiform()         // random author ordering
        sort _u
        gen int first_rnd = .
        *  Randomly assign timing to the first n_treated authors
        replace first_rnd = `T0' + floor((`N_MON' - `T0' + 1) * runiform()) ///
            if _n <= `n_treated'
        keep author_id first_rnd
        tempfile rnd_timing
        save `rnd_timing'
    restore

    merge m:1 author_id using `rnd_timing', nogen keep(1 3)

    * ── (4) Run both scenarios ────────────────────────────────────────────────
    foreach scen in det rnd {

        di as text "   scenario: `scen'"

        local treat_var = cond("`scen'" == "det", "first_det", "first_rnd")

        preserve    // ── preserve full panel ──────────────────────────

        * ── Build stacked dataset ────────────────────────────────────────────
        *
        *  cohort  = first_treat for treated authors
        *          = pseudo-cohort (uniform {T0..N_MON}) for never-treated
        *  Each author ends up in exactly ONE cohort stack.
        *  Never-treated whose pseudo-cohort has no treated authors are dropped.

        gen int  cohort       = `treat_var'       // . = never treated
        gen byte treated_unit = (cohort < .)

        *  Assign pseudo-cohort to never-treated (one draw per author, month 1)
        bysort author_id (month): replace cohort = ///
            `T0' + floor((`N_MON' - `T0' + 1) * runiform()) ///
            if treated_unit == 0 & _n == 1
        *  Broadcast cohort value to all months of the author
        bysort author_id (month): replace cohort = cohort[1]

        *  Drop cohorts with no treated unit (pseudo-only cohorts)
        bysort cohort: egen byte _has_tr = max(treated_unit)
        keep if _has_tr == 1
        drop _has_tr

        *  Relative event time
        gen int rel_time = month - cohort

        * ── FE variables ─────────────────────────────────────────────────────

        *  month_cohort_fe: calendar-month × cohort (unique cell identifier)
        *  cohort ∈ {11..30}, month ∈ {1..30}  →  no collision with *100 encoding
        gen long month_cohort_fe  = cohort * 100 + month

        * ── Event-study interaction dummies: I(rel_time=k) × treated ─────────
        *  Pre-period: rel_time ∈ {-W_PRE, …, -2}  (skip -1 = reference)
        forvalues k = 2 / `W_PRE' {
            local kstr = string(`k', "%02.0f")
            gen byte pre`kstr'_tr = (rel_time == -`k') * treated_unit
        }
        *  Treatment month: rel_time = 0
        gen byte time00_tr = (rel_time == 0) * treated_unit
        *  Post-period: rel_time ∈ {1, …, W_POST}
        forvalues k = 1 / `W_POST' {
            local kstr = string(`k', "%02.0f")
            gen byte post`kstr'_tr = (rel_time == `k') * treated_unit
        }

        * ── PPML regression ──────────────────────────────────────────────────

        quietly ppmlhdfe count pre*_tr time00_tr post*_tr,    ///
            absorb(author_id month_cohort_fe /*rel_time*/) ///
            vce(cluster author_id)

        * ── Store coefficients ───────────────────────────────────────────────
        *  Pre-period
        forvalues k = 2 / `W_PRE' {
            local kstr = string(`k', "%02.0f")
            post coef_collect ("`scen'") (`rep') (`= -`k'')   ///
                (_b[pre`kstr'_tr]) (_se[pre`kstr'_tr])
        }
        *  Reference period (zero by construction)
        post coef_collect ("`scen'") (`rep') (-1) (0) (0)
        *  Treatment month (rel_time = 0)
        post coef_collect ("`scen'") (`rep') (0)              ///
            (_b[time00_tr]) (_se[time00_tr])
        *  Post-period
        forvalues k = 1 / `W_POST' {
            local kstr = string(`k', "%02.0f")
            post coef_collect ("`scen'") (`rep') (`k')        ///
                (_b[post`kstr'_tr]) (_se[post`kstr'_tr])
        }

        restore     // ── restore full panel ──────────────────────────

    }   // end foreach scen (stacked PPML)

}   // end forvalues rep

postclose coef_collect
di as result _n "All replications complete."

* ─────────────────────────────────────────────────────────────────────────────
*  Aggregate and plot
* ─────────────────────────────────────────────────────────────────────────────
use "sim_stata_coefs_100K_1000.dta", clear

*  Monte-Carlo summary: mean ± 1.96 × (SD / sqrt(N_reps))
collapse                          ///
    (mean)  mean_est = estimate   ///
    (sd)    sd_est   = estimate   ///
    (count) n_reps   = estimate   ///
    , by(scenario rel_time)

gen double se_mean = sd_est / sqrt(n_reps)
gen double ci_lo   = mean_est - 1.96 * se_mean
gen double ci_hi   = mean_est + 1.96 * se_mean

save "sim_stata_coefs_summary_100K_1000.dta", replace

*  Colours matching simulation_selection.R:
*    detection → #dd8452 = RGB 221 132 82
*    random    → #8172b2 = RGB 129 114 178

twoway                                                                   ///
    (rcap   ci_lo ci_hi rel_time if scenario == "det",                   ///
        lcolor("221 132 82")  lwidth(thin))                              ///
    (scatter mean_est rel_time if scenario == "det",                     ///
        mcolor("221 132 82")  msymbol(circle) msize(small))             ///
    (rcap   ci_lo ci_hi rel_time if scenario == "rnd",                   ///
        lcolor("129 114 178") lwidth(thin))                              ///
    (scatter mean_est rel_time if scenario == "rnd",                     ///
        mcolor("129 114 178") msymbol(square) msize(small))             ///
    ,                                                                    ///
    yline(0, lcolor(black) lwidth(thin))                                 ///
    xline(-0.5, lcolor(gs8) lwidth(thin) lpattern(dash))                 ///
    xlabel(-`W_PRE'(2)`W_POST', labsize(small) nogrid)                    ///
    xtitle("Months relative to treatment") ytitle("Change in author productivity (%)") ///      
    legend(order(2 "IID productivity + first detection"                  ///
                 4 "IID productivity + random timing")                   ///
           position(12) rows(1) size(small))

graph export "Fig2_Simulation_Stacked_100K_1000.pdf", replace


* ─────────────────────────────────────────────────────────────────────────────
*  Rejection rates
*  "the t-test rejects H0: γ_k=0 in favor of a positive effect in X% of
*   cases for k∈[1,17] under detection, compared with Y% under random timing"
* ─────────────────────────────────────────────────────────────────────────────

use "sim_stata_coefs_100K_1000.dta", clear
keep if rel_time > 0

gen byte reject_k = (estimate / se > 1.645)
collapse (mean) pct_reject_k = reject_k, by(scenario)
format pct_reject_k %6.4f
di _n "Per-period rejection rate H0: γ_k=0 (one-sided 5%, k = 1..17):"
list scenario pct_reject_k, clean noobs

