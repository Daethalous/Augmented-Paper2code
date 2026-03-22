import json
import argparse
import copy

def extract_and_remove_bib(data, bib_data_list):
    # If data is a dictionary, recursively check its keys
    if isinstance(data, dict):
        # Extract bib_entries if present
        if "bib_entries" in data:
            bib_data_list.append(data.pop("bib_entries"))

        # Remove specific keys if present
        for key in ["ref_spans", "eq_spans", "authors", \
                    "year", "venue", "identifiers", "_pdf_hash", "header", "bib_entries"]:
            data.pop(key, None)
        # Recursively apply to child dictionaries or lists
        for key, value in data.items():
            data[key] = extract_and_remove_bib(value, bib_data_list)
    # If data is a list, apply the function to each item
    elif isinstance(data, list):
        return [extract_and_remove_bib(item, bib_data_list) for item in data]
    return data

def main(args):
    input_json_path = args.input_json_path
    output_json_path = args.output_json_path 
    output_bib_path = args.output_bib_path

    with open(f'{input_json_path}') as f:
        data = json.load(f)

    # Use a list to collect bib_entries found during recursion
    # Usually there is only one bib_entries dictionary in the whole file (under pdf_parse), 
    # but the recursive structure supports finding it anywhere.
    bib_data_list = []
    
    # We clarify that we're modifying 'data' in place, but let's be explicit
    cleaned_data = extract_and_remove_bib(data, bib_data_list)
    
    # Consolidate bib_entries. 
    # If multiple were found (unlikely for standard structure), we merge them or take the first one.
    # Based on standard structure, there's usually one big dictionary for bib_entries.
    if len(bib_data_list) > 0:
        final_bib_data = bib_data_list[0]
        # If there are more, we could merge them, but usually it's just one block.
        for extra_bib in bib_data_list[1:]:
            final_bib_data.update(extra_bib)
    else:
        final_bib_data = {}

    print(f"[SAVED] {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(cleaned_data, f)

    if output_bib_path:
        print(f"[SAVED] {output_bib_path}")
        with open(output_bib_path, 'w') as f:
            json.dump(final_bib_data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json_path", type=str)
    parser.add_argument("--output_json_path", type=str)
    parser.add_argument("--output_bib_path", type=str, help="Path to save the extracted bib_entries")

    args = parser.parse_args()
    main(args)
