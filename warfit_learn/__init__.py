import warnings
# Suppress a harmless sklearn warning or two
warnings.filterwarnings(
    action="ignore", module="sklearn", message="^internal gelsd"
)