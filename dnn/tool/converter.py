import xml.etree.ElementTree as ET
from datetime import datetime


def parse_datetime(dt_str):
    return datetime.strptime(dt_str.strip(), "%Y-%m-%dT%H:%M:%S.%f")


def convert_events_to_annotations(input_xml, output_file):
    tree = ET.ElementTree(ET.fromstring(input_xml))
    root = tree.getroot()

    annotationlist = ET.Element("annotationlist")

    # Extract GUID and ID from PatientInfo
    patient_info = root.find(".//PatientInfo")
    if patient_info is not None:
        guid = patient_info.find("GUID")
        id_ = patient_info.find("ID")
        if guid is not None:
            ET.SubElement(annotationlist, "GUID").text = guid.text.strip()
        if id_ is not None:
            ET.SubElement(annotationlist, "ID").text = id_.text.strip()

    first_event = root.find(".//Event/StartTime")
    if first_event is not None:
        start_time = first_event.text.strip()
        ET.SubElement(annotationlist, "recording_start_time").text = start_time

    valid_types = {"SLEEP-S0": "SLEEP-W", "SLEEP-S1": "SLEEP-S1", "SLEEP-S2": "SLEEP-S2", "SLEEP-S3": "SLEEP-S3",
                   "SLEEP-REM": "SLEEP-REM"}

    for event in root.findall(".//Event"):
        event_type = event.find("Type")
        if event_type is not None and event_type.text.strip() in valid_types:
            annotation = ET.SubElement(annotationlist, "annotation")

            onset = event.find("StartTime").text.strip()
            stop_time = event.find("StopTime").text.strip()
            duration = int((parse_datetime(stop_time) - parse_datetime(onset)).total_seconds())

            description = valid_types[event_type.text.strip()]

            ET.SubElement(annotation, "onset").text = onset
            ET.SubElement(annotation, "duration").text = str(duration)
            ET.SubElement(annotation, "description").text = description

    xml_bytes = ET.tostring(annotationlist, encoding="iso-8859-1", method="xml")
    with open(output_file, "wb") as f:
        f.write(xml_bytes)


# Example usage
with open("../4.uncompressed_data/1.first_phase/22cd54f1-5d96-4b12-a392-d5ddada9d8d3/615070_iso-8859-1.xml", "r", encoding="iso-8859-1") as file:
    xml_content = file.read()

output_file = "converted_annotations.xml"
convert_events_to_annotations(xml_content, output_file)
print(f"Converted XML written to {output_file}")
