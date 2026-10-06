import io
import zipfile


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 2_000


class WorkbookUploadError(ValueError):
    """Raised when an uploaded workbook is unsafe or not a supported XLSX file."""


def validate_xlsx_upload(file_bytes, filename=""):
    """Validate an XLSX container before it is handed to a spreadsheet parser."""
    if not file_bytes:
        raise WorkbookUploadError("The uploaded workbook is empty.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise WorkbookUploadError("The uploaded workbook exceeds the 10 MB limit.")
    if filename and not str(filename).lower().endswith(".xlsx"):
        raise WorkbookUploadError("Only .xlsx workbooks are accepted.")

    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            members = archive.infolist()
            member_names = {member.filename.replace("\\", "/") for member in members}

            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise WorkbookUploadError("The workbook contains too many internal files.")
            if sum(member.file_size for member in members) > MAX_UNCOMPRESSED_BYTES:
                raise WorkbookUploadError("The workbook expands beyond the 50 MB safety limit.")
            if "[Content_Types].xml" not in member_names or "xl/workbook.xml" not in member_names:
                raise WorkbookUploadError("The file is not a valid XLSX workbook.")
            if any(
                name.startswith("xl/externalLinks/") or name.endswith("vbaProject.bin")
                for name in member_names
            ):
                raise WorkbookUploadError(
                    "Workbooks containing external links or macros are not accepted."
                )
            if any(name.startswith("/") or "../" in name for name in member_names):
                raise WorkbookUploadError("The workbook contains an unsafe internal path.")
    except zipfile.BadZipFile as exc:
        raise WorkbookUploadError("The file is not a valid XLSX workbook.") from exc

    return io.BytesIO(file_bytes)
