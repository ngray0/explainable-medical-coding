from typing import Any, Optional

import torch
from transformers import PreTrainedTokenizer

from datasets import DatasetDict
from explainable_medical_coding.utils.settings import TARGET_COLUMN
from explainable_medical_coding.utils.datatypes import Lookups
from explainable_medical_coding.utils.tokenizer import TargetTokenizer


def create_lookups(
    dataset: DatasetDict,
    text_tokenizer: PreTrainedTokenizer,
    target_tokenizer: TargetTokenizer,
) -> Lookups:
    """Load the lookups.

    Args:
        dataframe (pd.DataFrame): The dataframe.
        tokenizer (PreTrainedTokenizer): The tokenizer.

    Returns:
        Lookups: The lookups.
    """

    split2code_indices = create_split2target_indices_lookup(
        dataset, target_tokenizer=target_tokenizer
    )
    data_info = get_data_info(
        dataset=dataset,
        vocab_size=len(text_tokenizer),
        pad_token_id=text_tokenizer.pad_token_id,
        pad_target_id=target_tokenizer.pad_id,
        sos_target_id=target_tokenizer.sos_id,
        eos_target_id=target_tokenizer.eos_id,
        num_classes=len(target_tokenizer),
        split2code_indices=split2code_indices,
    )

    # Create code system mappings for diagnosis vs procedure codes
    code_system2code_indices = create_code_system_mappings(dataset, target_tokenizer)
    if code_system2code_indices:
        data_info["code_system2code_indices"] = code_system2code_indices

    return Lookups(
        data_info=data_info,
        split2code_indices=split2code_indices,
        target_tokenizer=target_tokenizer,
    )


def create_split2target_indices_lookup(
    dataset: DatasetDict, target_tokenizer: TargetTokenizer
) -> dict[str, torch.Tensor]:
    split2code_indices = {}
    for split_name, data in dataset.items():
        unique_codes = (
            data.with_format("pandas")[TARGET_COLUMN].explode().unique().tolist()
        )
        target_ids = target_tokenizer(unique_codes)
        split2code_indices[split_name] = torch.tensor(target_ids, dtype=torch.long)
    split2code_indices["train_val"] = split2code_indices["train"]
    return split2code_indices


def get_data_info(
    dataset: DatasetDict,
    vocab_size: int,
    pad_token_id: int,
    pad_target_id: Optional[int],
    sos_target_id: Optional[int],
    eos_target_id: Optional[int],
    num_classes: int,
    split2code_indices: dict[str, list],
) -> dict:
    data_info: dict[str, Any] = {}
    data_info["num_examples"] = (
        len(dataset["train"]) + len(dataset["validation"]) + len(dataset["test"])
    )
    data_info["num_train_examples"] = len(dataset["train"])
    data_info["num_val_examples"] = len(dataset["validation"])
    data_info["num_test_examples"] = len(dataset["test"])

    data_info["average_words_per_example"] = (
        dataset["train"].with_format("pandas")["input_ids"].apply(len).mean()
        + dataset["validation"].with_format("pandas")["input_ids"].apply(len).mean()
        + dataset["test"].with_format("pandas")["input_ids"].apply(len).mean()
    ) / 3

    data_info["average_targets_per_example"] = (
        dataset["train"].with_format("pandas")[TARGET_COLUMN].apply(len).mean()
        + dataset["validation"].with_format("pandas")[TARGET_COLUMN].apply(len).mean()
        + dataset["test"].with_format("pandas")[TARGET_COLUMN].apply(len).mean()
    ) / 3

    data_info["num_classes"] = num_classes
    data_info["num_train_classes"] = len(split2code_indices["train"])
    data_info["num_val_classes"] = len(split2code_indices["validation"])
    data_info["num_test_classes"] = len(split2code_indices["test"])
    data_info["vocab_size"] = vocab_size
    data_info["pad_token_id"] = pad_token_id
    data_info["pad_target_id"] = pad_target_id
    data_info["sos_target_id"] = sos_target_id
    data_info["eos_target_id"] = eos_target_id

    return data_info


def create_code_system_mappings(dataset: DatasetDict, target_tokenizer: TargetTokenizer) -> dict[str, torch.Tensor]:
    """Create mappings for diagnosis vs procedure codes based on source columns."""
    code_system2code_indices = {}
    
    # Get available columns from the first split
    for split_name, data in dataset.items():
        available_columns = data.column_names
        break
    
    print(f"🔍 Available dataset columns: {available_columns}")
    
    # Look for diagnosis and procedure columns with flexible naming
    # Pattern: any column containing "diag" but not "type" maps to "diagnosis", same for "proc"
    diagnosis_columns = [col for col in available_columns if "diag" in col.lower() and "type" not in col.lower()]
    procedure_columns = [col for col in available_columns if "proc" in col.lower() and "type" not in col.lower()]
    
    print(f"🔍 Found diagnosis columns: {diagnosis_columns}")
    print(f"🔍 Found procedure columns: {procedure_columns}")
    
    if not (diagnosis_columns or procedure_columns):
        print("❌ No diagnosis/procedure columns found - using traditional evaluation")
        print("   (Looking for columns containing 'diag' or 'proc')")
        return {}
    
    # Create code system to code sets mapping
    code_system2codes = {}
    if diagnosis_columns:
        code_system2codes["diagnosis"] = set()
    if procedure_columns:
        code_system2codes["procedure"] = set()
    
    # Collect all codes by system across all splits
    for split_name, data in dataset.items():
        df = data.to_pandas()
        
        # Extract diagnosis codes from all diagnosis columns
        for diag_col in diagnosis_columns:
            if diag_col in df.columns:
                diag_codes = df[diag_col].explode().dropna().unique().tolist()
                code_system2codes["diagnosis"].update(diag_codes)
                print(f"✅ Added {len(diag_codes)} unique codes from {diag_col}")
        
        # Extract procedure codes from all procedure columns  
        for proc_col in procedure_columns:
            if proc_col in df.columns:
                proc_codes = df[proc_col].explode().dropna().unique().tolist()
                code_system2codes["procedure"].update(proc_codes)
                print(f"✅ Added {len(proc_codes)} unique codes from {proc_col}")
    
    # Convert to target indices for each code system
    for code_system, codes in code_system2codes.items():
        if codes:
            # Format codes with proper punctuation before tokenizer lookup
            from explainable_medical_coding.utils.data_helper_functions import reformat_icd9cm_code, reformat_icd9pcs_code, reformat_icd10cm_code
            
            formatted_codes = []
            for code in codes:
                if code_system == "diagnosis":
                    # Assume ICD-10 for diagnosis codes (adjust based on your dataset)
                    formatted_code = reformat_icd10cm_code(code)
                elif code_system == "procedure":
                    # Assume ICD-9 PCS for procedure codes (adjust based on your dataset)  
                    formatted_code = reformat_icd9pcs_code(code)
                else:
                    formatted_code = code
                formatted_codes.append(formatted_code)
            
            # Get indices for formatted codes that exist in the target tokenizer
            valid_codes = [code for code in formatted_codes if code in target_tokenizer.target2id]
            invalid_codes = [code for code in formatted_codes if code not in target_tokenizer.target2id]
            
            print(f"📋 {code_system}: {len(codes)} raw codes → {len(formatted_codes)} formatted → {len(valid_codes)} valid")
            if invalid_codes:
                print(f"   ❌ First 5 invalid codes: {invalid_codes[:5]}")
            
            if valid_codes:
                target_ids = target_tokenizer(valid_codes)
                code_system2code_indices[code_system] = torch.tensor(target_ids, dtype=torch.long)
                print(f"✅ Created {code_system} mapping with {len(target_ids)} codes")
    
    return code_system2code_indices
