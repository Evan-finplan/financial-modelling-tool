import io
import zipfile
import unittest

from openpyxl import Workbook

from upload_security import MAX_UPLOAD_BYTES, WorkbookUploadError, validate_xlsx_upload


def build_workbook_bytes():
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "inputs"
    worksheet.append(["input_name", "input_value"])
    worksheet.append(["projection_years", 40])
    output = io.BytesIO()
    workbook.save(output)
    return output.getvalue()


class UploadSecurityTests(unittest.TestCase):
    def test_accepts_valid_xlsx(self):
        result = validate_xlsx_upload(build_workbook_bytes(), "inputs.xlsx")
        self.assertEqual(result.read(2), b"PK")

    def test_rejects_non_xlsx_extension(self):
        for filename in ["inputs.xls", "inputs.xlsm", "inputs.csv"]:
            with self.subTest(filename=filename):
                with self.assertRaisesRegex(WorkbookUploadError, "Only .xlsx"):
                    validate_xlsx_upload(build_workbook_bytes(), filename)

    def test_rejects_invalid_container(self):
        with self.assertRaisesRegex(WorkbookUploadError, "not a valid XLSX"):
            validate_xlsx_upload(b"not a workbook", "inputs.xlsx")

    def test_rejects_oversized_upload(self):
        with self.assertRaisesRegex(WorkbookUploadError, "10 MB"):
            validate_xlsx_upload(b"x" * (MAX_UPLOAD_BYTES + 1), "inputs.xlsx")

    def test_rejects_external_links(self):
        source = zipfile.ZipFile(io.BytesIO(build_workbook_bytes()))
        output = io.BytesIO()
        with source, zipfile.ZipFile(output, "w") as target:
            for member in source.infolist():
                target.writestr(member, source.read(member.filename))
            target.writestr("xl/externalLinks/externalLink1.xml", "<externalLink />")

        with self.assertRaisesRegex(WorkbookUploadError, "external links or macros"):
            validate_xlsx_upload(output.getvalue(), "inputs.xlsx")


if __name__ == "__main__":
    unittest.main()
