* =============================================================================
* Run regression + twoway plot for one dataset
* =============================================================================

capture program drop run_analysis
program define run_analysis
    args csv_in fig_out

    import delimited "`csv_in'", clear

    ppmlhdfe monthly_productivity rel_month_p*, ///
        absorb(hashed_author cohort##month) vce(cluster hashed_author)

    * Extract coefficients: pre(-12..-2), ref(-1)=0, post(0..17)
    * 11 pre + 1 ref + 18 post = 30 rows
    matrix coefs = J(30, 3, .)
    local row = 1
    forvalues k = 12(-1)2 {
        local kstr = string(`k', "%02.0f")
        matrix coefs[`row', 1] = -`k'
        matrix coefs[`row', 2] = _b[rel_month_pre_`kstr'_treated]
        matrix coefs[`row', 3] = _se[rel_month_pre_`kstr'_treated]
        local row = `row' + 1
    }
    matrix coefs[`row', 1] = -1
    matrix coefs[`row', 2] = 0
    matrix coefs[`row', 3] = 0
    local row = `row' + 1
    forvalues k = 0/17 {
        local kstr = string(`k', "%02.0f")
        matrix coefs[`row', 1] = `k'
        matrix coefs[`row', 2] = _b[rel_month_post_`kstr'_treated]
        matrix coefs[`row', 3] = _se[rel_month_post_`kstr'_treated]
        local row = `row' + 1
    }

    clear
    svmat double coefs
    rename (coefs1 coefs2 coefs3) (rel_time estimate se)
    gen double ci_lo = estimate - 1.96 * se
    gen double ci_hi = estimate + 1.96 * se

    twoway ///
        (rcap   ci_lo ci_hi rel_time, lcolor("33 102 172") lwidth(thin)) ///
        (scatter estimate  rel_time, mcolor("33 102 172") msymbol(circle) msize(small)) ///
        , yline(0, lcolor(black) lwidth(thin)) ///
        xline(-0.5, lcolor(gs8) lwidth(thin) lpattern(dash)) ///
        xlabel(-12(2)17, labsize(small) nogrid) ///
        xtitle("Months relative to first adoption") ///
        ytitle("Change in author productivity (%)") ///
        legend(off)

    graph export "`fig_out'", replace
end


* =============================================================================
* Hrun regression + save coefficient dataset (for multi-series plots)
* =============================================================================

capture program drop collect_coefs
program define collect_coefs
    args csv_in lbl outfile

    import delimited "`csv_in'", clear

    ppmlhdfe monthly_productivity rel_month_p*, ///
        absorb(hashed_author cohort##month) vce(cluster hashed_author)

    matrix coefs = J(30, 3, .)
    local row = 1
    forvalues k = 12(-1)2 {
        local kstr = string(`k', "%02.0f")
        matrix coefs[`row', 1] = -`k'
        matrix coefs[`row', 2] = _b[rel_month_pre_`kstr'_treated]
        matrix coefs[`row', 3] = _se[rel_month_pre_`kstr'_treated]
        local row = `row' + 1
    }
    matrix coefs[`row', 1] = -1
    matrix coefs[`row', 2] = 0
    matrix coefs[`row', 3] = 0
    local row = `row' + 1
    forvalues k = 0/17 {
        local kstr = string(`k', "%02.0f")
        matrix coefs[`row', 1] = `k'
        matrix coefs[`row', 2] = _b[rel_month_post_`kstr'_treated]
        matrix coefs[`row', 3] = _se[rel_month_post_`kstr'_treated]
        local row = `row' + 1
    }

    clear
    svmat double coefs
    rename (coefs1 coefs2 coefs3) (rel_time estimate se)
    gen double ci_lo = estimate - 1.96 * se
    gen double ci_hi = estimate + 1.96 * se
    gen str20 series = "`lbl'"
    save "`outfile'", replace
end


* =============================================================================
* Run all analyses
* =============================================================================

run_analysis ///
    "stacked_replication.csv" ///
    "Fig3_Replication.pdf"
	
summarize estimate if rel_time > 0

* ── Combined keyword-placebo figure ─────────────────────────────────────────
tempfile kw_data kw_paper kw_find
collect_coefs "stacked_placebo_keyword_data.csv"       "data"  `kw_data'
collect_coefs "stacked_placebo_keyword_paper.csv" "paper" `kw_paper'
collect_coefs "stacked_placebo_keyword_find.csv"  "find"  `kw_find'

use `kw_data', clear
append using `kw_paper'
append using `kw_find'

summarize estimate if rel_time > 0
sort series
by series: summarize estimate if rel_time > 0

twoway ///
    (rcap    ci_lo ci_hi rel_time if series == "data",  lcolor("33 102 172") lwidth(thin)) ///
    (scatter estimate  rel_time if series == "data",    mcolor("33 102 172") msymbol(circle)   msize(small)) ///
    (rcap    ci_lo ci_hi rel_time if series == "paper", lcolor("214 96 77")  lwidth(thin)) ///
    (scatter estimate  rel_time if series == "paper",   mcolor("214 96 77")  msymbol(square)   msize(small)) ///
    (rcap    ci_lo ci_hi rel_time if series == "find",  lcolor("77 172 38")  lwidth(thin)) ///
    (scatter estimate  rel_time if series == "find",    mcolor("77 172 38")  msymbol(triangle) msize(small)) ///
    , yline(0, lcolor(black) lwidth(thin)) ///
    xline(-0.5, lcolor(gs8) lwidth(thin) lpattern(dash)) ///
    xlabel(-12(2)17, labsize(small) nogrid) ///
    xtitle("Months relative to first adoption") ///
    ytitle("Change in author productivity (%)") ///
    legend(order(2 "data" 4 "paper" 6 "find") rows(1) position(12))
graph export "Fig3_Placebo_Keywords.pdf", replace


* ── Combined random-placebo figure ──────────────────────────────────────────
tempfile rnd10 rnd20 rnd30
collect_coefs "stacked_placebo_random10.csv" "p = 0.1" `rnd10'
collect_coefs "stacked_placebo_random20.csv" "p = 0.2" `rnd20'
collect_coefs "stacked_placebo_random30.csv" "p = 0.3" `rnd30'

use `rnd10', clear
append using `rnd20'
append using `rnd30'

summarize estimate if rel_time > 0
sort series
by series: summarize estimate if rel_time > 0

twoway ///
    (rcap    ci_lo ci_hi rel_time if series == "p = 0.1", lcolor("33 102 172") lwidth(thin)) ///
    (scatter estimate  rel_time if series == "p = 0.1",   mcolor("33 102 172") msymbol(circle)   msize(small)) ///
    (rcap    ci_lo ci_hi rel_time if series == "p = 0.2", lcolor("214 96 77")  lwidth(thin)) ///
    (scatter estimate  rel_time if series == "p = 0.2",   mcolor("214 96 77")  msymbol(square)   msize(small)) ///
    (rcap    ci_lo ci_hi rel_time if series == "p = 0.3", lcolor("77 172 38")  lwidth(thin)) ///
    (scatter estimate  rel_time if series == "p = 0.3",   mcolor("77 172 38")  msymbol(triangle) msize(small)) ///
    , yline(0, lcolor(black) lwidth(thin)) ///
    xline(-0.5, lcolor(gs8) lwidth(thin) lpattern(dash)) ///
    xlabel(-12(2)17, labsize(small) nogrid) ///
    xtitle("Months relative to first adoption") ///
    ytitle("Change in author productivity (%)") ///
    legend(order(2 "p = 0.1" 4 "p = 0.2" 6 "p = 0.3") rows(1) position(12))
graph export "Fig3_Placebo_Random.pdf", replace


run_analysis ///
    "stacked_placebo_prechatgpt.csv" ///
    "Fig3_Placebo_Pre_Period.pdf"

summarize estimate if rel_time > 0
	