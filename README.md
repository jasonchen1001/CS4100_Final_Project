# CS4100_Final_Project (Grocery List Optimizer)
Team Member: Kashish Sethi, Yanzhen Chen, Rongxuan Zhang

## Overview

Grocery shopping is a frequent and essential task that directly affects both financial and physical well-being. Many people struggle to balance food cost, nutritional needs, and dietary restrictions, often resulting in unhealthy or cost-inefficient choices. This challenge is particularly relevant for college students and young adults who face tighter budgets while trying to maintain fitness and health goals.

This project aims to build a **grocery list optimizer** that generates cost-effective grocery recommendations while satisfying users’ nutritional, dietary, and financial constraints. Rather than producing generic meal plans, the system is designed to generate realistic grocery lists that align with both user goals and past purchasing behavior.

---

## Problem Statement

The main goal of this project is to automatically generate a grocery list that:

* Meets user-specified nutritional targets based on physical profile and health goals
* Stays within a user-defined grocery budget
* Accounts for dietary constraints (e.g., vegetarian diets, food allergies)
* Remains consistent with user preferences inferred from historical purchase behavior

The challenge lies in balancing these competing constraints within a large and complex search space, where many possible grocery combinations may satisfy some but not all requirements.

---

## User Inputs

The system accepts the following user inputs:

* Basic profile information (e.g., height, weight, age, activity level)
* Health or fitness goals
* Weekly or monthly grocery budget
* Dietary constraints and restrictions
* Historical grocery purchase data (if available)

These inputs are used to guide both nutritional targets and personalization of the generated grocery list.

---

## Data Assumptions

To ensure reproducibility and maintain focus on optimization rather than data collection, the project uses a **curated static table** of grocery items. Each item includes average supermarket pricing and basic nutrition information.

If time permits, an optional script may be added to refresh a cached snapshot of this data from public sources. However, the core optimization pipeline does **not** depend on live APIs or real-time pricing.

---

## Related Work

One existing approach to automated meal planning is the website **Eat This Much**(https://www.eatthismuch.com/
), which generates daily meal plans based on dietary restrictions and calorie goals. However, this approach does not account for grocery budgets or cost constraints, which is a central focus of this project.

---

## Proposed Approach

While the specific AI method has not yet been finalized, one potential direction is to frame the problem as a constrained optimization task. Given the large search space and multiple competing constraints, heuristic or evolutionary approaches (such as genetic algorithms) may be appropriate.

In such a formulation:

* A candidate solution represents a grocery list
* Each gene corresponds to the quantity of a grocery item
* A fitness function evaluates how well a solution satisfies nutritional goals, budget constraints, and preference consistency

The final approach will be selected based on feasibility and performance during development.

## Project Status

This repository is under active development. Current work focuses on problem formulation, data schema design, and baseline methods before implementing and evaluating optimization algorithms.

