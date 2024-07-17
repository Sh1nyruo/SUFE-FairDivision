# Fair Division with Indivisible Items

This project implements algorithms to find Envy-Free up to any item (EFX) allocations of indivisible items with donations. The goal is to find an EFX allocation or maximize the Nash Social Welfare (NSW).

## Algorithm Theory

The theory behind this algorithm is derived from the article: [Envy-freeness up to any item with high Nash welfare: The virtue of donating items](https://arxiv.org/abs/1902.04319).

## Base Class

The base class of this algorithm is adapted from the GitHub library: [erelsgl/fairpy](https://github.com/erelsgl/fairpy).

## Implemented Algorithms

This repository contains implementations of algorithms to find allocations based on:
- Maximum Nash Social Welfare (NSW)
- EFX (Envy-Free up to any item) criteria

## Usage

Please refer to the `FairDivision.md` file to see how to use these algorithms in detail.

## Requirements

- Python version: 3.10
- All required libraries are listed in `requirements.txt`.

## Installation

To install the required libraries, run:

```bash
pip install -r requirements.txt
