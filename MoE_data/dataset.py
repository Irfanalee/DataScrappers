"""
Dataset Downloader for Multi-Modal Document Analyzer
Downloads and prepares datasets for fine-tuning.

Datasets:
1. DocVQA - Document Visual QA (general documents) -- done
2. Docmatix - Large-scale DocVQA (2.4M images, 9.5M Q&A pairs)
3. CORD - Receipt/Invoice parsing -- done
4. CUAD - Contract Understanding (legal contracts) -- done
5. SROIE - Scanned Receipts OCR -- done

Usage:
    python dataset.py --all
    python dataset.py --dataset docvqa
    python dataset.py --dataset cord
"""

import argparse
import json
import shutil
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

OUTPUT_DIR = "data"


def download_docvqa():
    """
    Download DocVQA dataset.
    - 50,000 questions on 12,767 document images
    - Mix of printed, typewritten, handwritten content
    - Letters, memos, notes, reports
    """
    print("\n" + "=" * 60)
    print("Downloading DocVQA...")
    print("=" * 60)
    
    output_path = Path(OUTPUT_DIR) / "docvqa"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load from HuggingFace
    dataset = load_dataset("HuggingFaceM4/DocumentVQA")
    
    print(f"Train: {len(dataset['train'])} examples")
    print(f"Validation: {len(dataset['validation'])} examples")
    print(f"Test: {len(dataset['test'])} examples")
    
    # Save
    dataset.save_to_disk(str(output_path))
    print(f"Saved to: {output_path}")
    
    # Preview
    print("\nSample:")
    sample = dataset['train'][0]
    print(f"  Question: {sample.get('question', sample.get('questions', ['N/A'])[0] if isinstance(sample.get('questions'), list) else 'N/A')}")
    print(f"  Keys: {list(sample.keys())}")
    
    return dataset


def download_docmatix():
    """
    Download Docmatix dataset (subset).
    - 2.4 million images, 9.5 million Q/A pairs
    - 100x larger than DocVQA
    - We download a subset for fine-tuning
    """
    print("\n" + "=" * 60)
    print("Downloading Docmatix (subset)...")
    print("=" * 60)
    
    output_path = Path(OUTPUT_DIR) / "docmatix"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load subset (full dataset is huge)
    # Using streaming to avoid downloading everything
    dataset = load_dataset(
        "HuggingFaceM4/Docmatix",
        "images",
        split="train",
        streaming=True
    )

    # Save incrementally to avoid OOM — images go to disk, metadata to JSONL
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = output_path / "docmatix_50k.jsonl"

    max_examples = 50000
    print(f"Downloading first {max_examples} examples (streaming to disk)...")

    count = 0
    with open(metadata_file, "w") as f:
        for example in tqdm(dataset, total=max_examples):
            if count >= max_examples:
                break

            # Save images to disk
            saved_images = []
            images = example.get("images", [])
            if not isinstance(images, list):
                images = [images]
            for img_idx, img in enumerate(images):
                img_path = images_dir / f"{count:06d}_{img_idx}.png"
                if hasattr(img, "save"):
                    img.save(str(img_path))
                    saved_images.append(str(img_path))

            # Write metadata (without image data) as a JSONL line
            meta = {k: v for k, v in example.items() if k != "images"}
            meta["image_paths"] = saved_images
            meta["example_id"] = count
            f.write(json.dumps(meta) + "\n")

            count += 1

    print(f"Downloaded: {count} examples")
    print(f"Images saved to: {images_dir}")
    print(f"Metadata saved to: {metadata_file}")

    return count


def download_cord():
    """
    Download CORD dataset (receipts/invoices).
    - 11,000+ Indonesian receipts
    - 30 semantic classes, 5 superclasses
    - Menu items, totals, subtotals
    """
    print("\n" + "=" * 60)
    print("Downloading CORD (Receipt Dataset)...")
    print("=" * 60)
    
    output_path = Path(OUTPUT_DIR) / "cord"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Try different versions
    try:
        dataset = load_dataset("naver-clova-ix/cord-v2")
    except Exception:
        try:
            dataset = load_dataset("katanaml/cord")
        except Exception:
            dataset = load_dataset("Voxel51/consolidated_receipt_dataset")
    
    print(f"Splits: {list(dataset.keys())}")
    for split in dataset.keys():
        print(f"  {split}: {len(dataset[split])} examples")
    
    # Save
    dataset.save_to_disk(str(output_path))
    print(f"Saved to: {output_path}")
    
    # Preview
    print("\nSample:")
    sample = dataset[list(dataset.keys())[0]][0]
    print(f"  Keys: {list(sample.keys())}")
    
    return dataset


def download_cuad():
    """
    Download CUAD dataset (contracts).
    - 13,000+ labels in 510 commercial contracts
    - 41 categories of important clauses
    - Legal contract review
    """
    print("\n" + "=" * 60)
    print("Downloading CUAD (Contract Dataset)...")
    print("=" * 60)
    
    output_path = Path(OUTPUT_DIR) / "cuad"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load from HuggingFace (script-free parquet version)
    dataset = load_dataset("dvgodoy/CUAD_v1_Contract_Understanding_clause_classification")

    print(f"Splits: {list(dataset.keys())}")
    for split in dataset.keys():
        print(f"  {split}: {len(dataset[split])} examples")
    
    # Save
    dataset.save_to_disk(str(output_path))
    print(f"Saved to: {output_path}")
    
    # Preview
    print("\nSample:")
    sample = dataset[list(dataset.keys())[0]][0]
    print(f"  Keys: {list(sample.keys())}")

    return dataset


def download_sroie():
    """
    Download SROIE dataset (scanned receipts).
    - 973 scanned receipts in English
    - OCR and information extraction
    """
    print("\n" + "=" * 60)
    print("Downloading SROIE (Scanned Receipts)...")
    print("=" * 60)
    
    output_path = Path(OUTPUT_DIR) / "sroie"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # darentang/sroie is small (~973 receipts), safe to load directly
    # Only use streaming for the larger fallback dataset
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    dataset = None
    try:
        print("Trying darentang/sroie...")
        dataset = load_dataset("darentang/sroie")
    except Exception as e:
        print(f"  Failed: {e}")
        print("Trying fallback: priyank-m/SROIE_2019_text_recognition (streaming, capped at 2000)...")

    if dataset is not None:
        # Direct load succeeded — save splits to disk
        total_count = 0
        for split in dataset.keys():
            split_dir = images_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            metadata_file = output_path / f"sroie_{split}.jsonl"

            count = 0
            with open(metadata_file, "w") as f:
                for example in tqdm(dataset[split], desc=f"  {split}"):
                    saved_images = []
                    for key in ["image", "images"]:
                        img = example.get(key)
                        if img is None:
                            continue
                        imgs = img if isinstance(img, list) else [img]
                        for img_idx, im in enumerate(imgs):
                            img_path = split_dir / f"{count:06d}_{img_idx}.png"
                            if hasattr(im, "save"):
                                im.save(str(img_path))
                                saved_images.append(str(img_path))

                    meta = {k: v for k, v in example.items() if k not in ("image", "images")}
                    meta["image_paths"] = saved_images
                    meta["example_id"] = count
                    f.write(json.dumps(meta) + "\n")
                    count += 1

            print(f"  {split}: {count} examples")
            total_count += count
    else:
        # Fallback: stream with a cap to avoid 30K+ slow download
        max_examples = 2000
        total_count = 0
        for split in ["train", "test"]:
            try:
                stream = load_dataset(
                    "priyank-m/SROIE_2019_text_recognition",
                    split=split, streaming=True
                )
            except Exception:
                print(f"  Skipping split '{split}' (not found)")
                continue

            split_dir = images_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            metadata_file = output_path / f"sroie_{split}.jsonl"

            count = 0
            with open(metadata_file, "w") as f:
                for example in tqdm(stream, desc=f"  {split}", total=max_examples):
                    if count >= max_examples:
                        break
                    saved_images = []
                    for key in ["image", "images"]:
                        img = example.get(key)
                        if img is None:
                            continue
                        imgs = img if isinstance(img, list) else [img]
                        for img_idx, im in enumerate(imgs):
                            img_path = split_dir / f"{count:06d}_{img_idx}.png"
                            if hasattr(im, "save"):
                                im.save(str(img_path))
                                saved_images.append(str(img_path))

                    meta = {k: v for k, v in example.items() if k not in ("image", "images")}
                    meta["image_paths"] = saved_images
                    meta["example_id"] = count
                    f.write(json.dumps(meta) + "\n")
                    count += 1

            print(f"  {split}: {count} examples")
            total_count += count

    print(f"Downloaded: {total_count} total examples")
    print(f"Images saved to: {images_dir}")
    print(f"Saved to: {output_path}")

    return total_count


def download_funsd():
    """
    Download FUNSD dataset (forms).
    - Form understanding in noisy scanned documents
    - Entity and relation labeling
    """
    print("\n" + "=" * 60)
    print("Downloading FUNSD (Forms Dataset)...")
    print("=" * 60)
    
    output_path = Path(OUTPUT_DIR) / "funsd"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load from HuggingFace
    dataset = load_dataset("nielsr/funsd")
    
    print(f"Train: {len(dataset['train'])} examples")
    print(f"Test: {len(dataset['test'])} examples")
    
    # Save
    dataset.save_to_disk(str(output_path))
    print(f"Saved to: {output_path}")
    
    return dataset


def show_dataset_summary():
    """Show summary of available datasets."""
    print("\n" + "=" * 60)
    print("AVAILABLE DATASETS FOR DOCUMENT INTELLIGENCE")
    print("=" * 60)
    
    datasets_info = [
        {
            "name": "DocVQA",
            "id": "docvqa",
            "description": "Document Visual QA - general documents",
            "size": "50K questions, 12K images",
            "disk_gb": 5.0,
            "use_for": "General document understanding"
        },
        {
            "name": "Docmatix",
            "id": "docmatix",
            "description": "Large-scale DocVQA dataset (50K subset)",
            "size": "2.4M images, 9.5M Q&A pairs (downloading 50K subset)",
            "disk_gb": 15.0,
            "use_for": "Pre-training / large-scale fine-tuning"
        },
        {
            "name": "CORD",
            "id": "cord",
            "description": "Receipt parsing dataset",
            "size": "11K+ receipts",
            "disk_gb": 1.5,
            "use_for": "Invoice/Receipt extraction"
        },
        {
            "name": "CUAD",
            "id": "cuad",
            "description": "Contract Understanding dataset",
            "size": "510 contracts, 13K labels, 41 clause types",
            "disk_gb": 0.5,
            "use_for": "Contract analysis"
        },
        {
            "name": "SROIE",
            "id": "sroie",
            "description": "Scanned Receipts OCR",
            "size": "973 receipts",
            "disk_gb": 0.3,
            "use_for": "Receipt OCR + extraction"
        },
        {
            "name": "FUNSD",
            "id": "funsd",
            "description": "Form Understanding in Scanned Documents",
            "size": "199 forms",
            "disk_gb": 0.2,
            "use_for": "Form field extraction"
        },
    ]
    
    for ds in datasets_info:
        print(f"\n📄 {ds['name']} ({ds['id']})")
        print(f"   {ds['description']}")
        print(f"   Size: {ds['size']}")
        print(f"   Estimated disk space: ~{ds['disk_gb']} GB")
        print(f"   Use for: {ds['use_for']}")

    total_gb = sum(ds["disk_gb"] for ds in datasets_info)
    print(f"\nTotal estimated disk space (all datasets): ~{total_gb:.1f} GB")

    print("\n" + "-" * 60)
    print("Usage:")
    print("  python dataset.py --all              # Download all")
    print("  python dataset.py --dataset docvqa   # Download specific")
    print("  python dataset.py --dataset cord cuad # Download multiple")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for document intelligence")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--dataset", nargs="+", 
                        choices=["docvqa", "docmatix", "cord", "cuad", "sroie", "funsd"],
                        help="Specific dataset(s) to download")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    
    args = parser.parse_args()
    
    if args.list or (not args.all and not args.dataset):
        show_dataset_summary()
        return
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    download_functions = {
        "docvqa": download_docvqa,
        "docmatix": download_docmatix,
        "cord": download_cord,
        "cuad": download_cuad,
        "sroie": download_sroie,
        "funsd": download_funsd,
    }
    
    if args.all:
        datasets_to_download = list(download_functions.keys())
    else:
        datasets_to_download = args.dataset

    # Estimate disk space needed
    disk_estimates = {
        "docvqa": 5.0, "docmatix": 15.0, "cord": 1.5,
        "cuad": 0.5, "sroie": 0.3, "funsd": 0.2,
    }
    total_needed = sum(disk_estimates.get(d, 1.0) for d in datasets_to_download)
    free_gb = shutil.disk_usage(Path(OUTPUT_DIR).resolve()).free / (1024 ** 3)

    print("=" * 60)
    print("DATASET DOWNLOADER")
    print("=" * 60)
    print(f"Downloading: {', '.join(datasets_to_download)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nEstimated disk space needed: ~{total_needed:.1f} GB")
    for d in datasets_to_download:
        print(f"  - {d}: ~{disk_estimates.get(d, 1.0):.1f} GB")
    print(f"Available disk space: {free_gb:.1f} GB")

    if total_needed > free_gb:
        print(f"\n⚠️  WARNING: Not enough disk space! Need ~{total_needed:.1f} GB but only {free_gb:.1f} GB available.")
        return

    confirm = input(f"\nProceed with download (~{total_needed:.1f} GB)? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Download cancelled.")
        return

    for dataset_name in datasets_to_download:
        try:
            download_functions[dataset_name]()
        except Exception as e:
            print(f"\n❌ Error downloading {dataset_name}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Datasets saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()