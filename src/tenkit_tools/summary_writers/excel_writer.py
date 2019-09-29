import xlsxwriter

from ..utils import load_evaluations, load_summary


def _write_summary_excel(sheet, row, summary):
    sheet.write(row, 0, "Summary: ")
    row += 1
    for key, value in summary.items():
        sheet.write(row, 1, key)
        sheet.write(row, 2, str(value))
        row += 1
    row += 1
    return row


def _write_best_run_evaluation_excel(sheet, row, best_run_evaluations):
    sheet.write(row, 0, "Best run metrics:")
    row += 1
    for evaluation in best_run_evaluations:
        for metric_name, metric in evaluation.items():
            sheet.write(row, 1, metric_name)
            sheet.write(row, 2, metric)
            row += 1

    row += 5
    return row


def _write_multi_run_evaluation_excel(sheet, row, multi_run_evaluations):
    for eval_name, evaluations in multi_run_evaluations.items():
        sheet.write(row, 0, eval_name)
        row += 1
        col = 1
        for col_name, col_values in evaluations.items():
            sheet.write(row, col, col_name)
            row_modifier = 1
            for value in col_values:
                sheet.write(row + row_modifier, col, value)
                row_modifier += 1

            col += 1

        row += row_modifier + 2
    return row


def _write_visualisations_excel(sheet, row, figure_path):
    for figure in figure_path.glob("*.png"):
        sheet.insert_image(row, 0, figure)
        row += 40
    return row


def _create_spreadsheet(
    experiment_path, summary, best_run_evaluations, multi_run_evaluations
):
    print(
        "Storing summary sheet in: ", experiment_path / "summaries" / "evaluation.xslx"
    )
    figure_path = experiment_path / "summaries" / "visualizations"

    book = xlsxwriter.Workbook(experiment_path / "summaries" / "evaluation.xslx", {'nan_inf_to_errors': True})
    sheet = book.add_worksheet()
    fig_sheet = book.add_worksheet("Figures")

    row = _write_summary_excel(sheet, 0, summary)
    row = _write_best_run_evaluation_excel(sheet, row, best_run_evaluations)
    row = _write_multi_run_evaluation_excel(sheet, row, multi_run_evaluations)

    _write_visualisations_excel(fig_sheet, 0, figure_path)

    book.close()


def create_spreadsheet(experiment_path):
    summary = load_summary(experiment_path)
    evaluations = load_evaluations(experiment_path)
    _create_spreadsheet(experiment_path, summary=summary, **evaluations)
