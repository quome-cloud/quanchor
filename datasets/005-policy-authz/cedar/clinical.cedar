// Clinical authorization (Cedar). Context = { args, attrs } from the PIP.
// Prescribers may order medication during an active encounter, on a paneled
// patient, within the formulary dose ceiling.
permit(
  principal == Role::"prescriber",
  action == Action::"order_medication",
  resource
) when {
  context.attrs.active_encounter == true &&
  context.attrs.paneled == true &&
  context.args.dose_mg <= context.attrs.max_dose_mg
};

// Nurses may administer and view.
permit(principal == Role::"nurse", action == Action::"administer", resource);
permit(principal == Role::"nurse", action == Action::"view_record", resource);
permit(principal == Role::"prescriber", action == Action::"view_record", resource);
permit(principal == Role::"read_only", action == Action::"view_record", resource);
