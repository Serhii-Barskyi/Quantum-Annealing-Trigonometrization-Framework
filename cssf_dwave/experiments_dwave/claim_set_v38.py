"""Frozen v38 external comparison-set semantics for D-Wave evidence."""
PRIMARY_EXTERNAL_COMPARATORS=(
    "GP+EI-full",
    "Finzgar-BO-matched-full",
    "TuRBO-matched-full",
    "QZero-matched-full",
    "Worldline-Susceptibility-full",
    "Strong-SA",
    "Strong-Tabu",
)
SURROGATE_STRUCTURE_REFERENCES=("Periodic-GP","Torus-Riemannian-Matern-GP")
QUALITY_REFERENCE=("HiGHS-quality-reference",)
CLAIM_METHOD="CSSF-full"
__all__=["PRIMARY_EXTERNAL_COMPARATORS","SURROGATE_STRUCTURE_REFERENCES","QUALITY_REFERENCE","CLAIM_METHOD"]
