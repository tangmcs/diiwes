# Report index

Reports are compact review artifacts derived from validated local results. Raw
training histories and scheduler logs remain under the ignored `results/` and
`job_outputs/` trees.

## Mentor-requested no-trust-region comparison

- [`hopper_hessian_no_trust/mentor_report.html`](hopper_hessian_no_trust/mentor_report.html)
  is the self-contained technical report comparing Standard ES with the signed
  diagonal frozen-rank curvature surrogate across decreasing learning-rate
  sequences.
- [`hopper_hessian_no_trust/mentor_report_artifact.json`](hopper_hessian_no_trust/mentor_report_artifact.json)
  contains the report's machine-readable data and provenance.
- [`hopper_hessian_no_trust/source_notes.md`](hopper_hessian_no_trust/source_notes.md)
  records the validated inputs and interpretation boundaries.

The report embeds its charts and validated comparison data, so it can be
reviewed without copying the multi-gigabyte local result tree into Git.
