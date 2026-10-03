package authz

default allow := false

# Prescribers may order medication during an active encounter, on a paneled
# patient, within the formulary dose ceiling.
allow if {
    input.principal == "prescriber"
    input.action == "order_medication"
    input.attrs.active_encounter == true
    input.attrs.paneled == true
    input.args.dose_mg <= input.attrs.max_dose_mg
}

# Nurses may administer.
allow if {
    input.principal == "nurse"
    input.action == "administer"
}

# Nurses may view records.
allow if {
    input.principal == "nurse"
    input.action == "view_record"
}

# Prescribers may view records.
allow if {
    input.principal == "prescriber"
    input.action == "view_record"
}

# Read-only role may view records.
allow if {
    input.principal == "read_only"
    input.action == "view_record"
}
