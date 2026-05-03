* ─────────────────────────────────────────────────────────────────────────────
*  Fig1_Stacked: stacked PPML (author + cohort×month)
* ─────────────────────────────────────────────────────────────────────────────
import delimited "production_arxiv.csv", clear
ppmlhdfe monthly_productivity rel_month_p*, absorb(hashed_author cohort##month) vce(cluster hashed_author)

* Extract coefficients
tempfile coefs_stacked
postfile handle_stacked int rel_time double estimate se using `coefs_stacked', replace
forvalues k = 12(-1)2 {
    local kstr = string(`k', "%02.0f")
    post handle_stacked (`= -`k'') (_b[rel_month_pre_`kstr'_treated]) (_se[rel_month_pre_`kstr'_treated])
}
post handle_stacked (-1) (0) (0)    // reference period
forvalues k = 0/17 {
    local kstr = string(`k', "%02.0f")
    post handle_stacked (`k') (_b[rel_month_post_`kstr'_treated]) (_se[rel_month_post_`kstr'_treated])
}
postclose handle_stacked

use `coefs_stacked', clear
gen double ci_lo = estimate - 1.96 * se
gen double ci_hi = estimate + 1.96 * se

twoway ///
    (rcap   ci_lo ci_hi rel_time, lcolor("33 102 172") lwidth(thin)) ///
    (scatter estimate  rel_time, mcolor("33 102 172") msymbol(circle) msize(small)) ///
    , yline(0, lcolor(black) lwidth(thin)) ///
    xline(-0.5, lcolor(gs8) lwidth(thin) lpattern(dash)) ///
    xlabel(-12(2)17, labsize(small) nogrid) ///
    xtitle("Months relative to first adoption") ytitle("Change in author productivity (%)") ///
    legend(off)
graph export "Fig1_Stacked.pdf", replace
