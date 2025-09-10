# Student Simulator Documentation

Welcome to the Student Simulator documentation! This collection of guides will help you understand and use the simulator effectively.

## Quick Start

If you're new to the Student Simulator, start here:

1. **[Getting Started](getting-started.md)** - Basic concepts and a simple example
2. **[Advanced Simulation](advanced-simulation.md)** - Large-scale simulations with complex skill hierarchies

## What is the Student Simulator?

The Student Simulator is a Python library that models how students learn skills over time. It's designed for:

- **Educational researchers** studying learning progression
- **Assessment developers** testing item validity
- **Data scientists** analyzing educational data
- **Curriculum designers** understanding skill dependencies

## Key Features

- **Skill-based learning model** with prerequisite relationships
- **Realistic learning parameters** based on educational research
- **Flexible assessment generation** with customizable item pools
- **Student heterogeneity** with random skill initialization
- **Visualization tools** for skill dependencies and mastery
- **CSV export** for analysis in other tools

## Example Use Cases

- **Learning Analytics**: Track how students progress through skill hierarchies
- **Assessment Design**: Test the validity of new assessment items
- **Curriculum Planning**: Understand which skills are prerequisites for others
- **Educational Research**: Study learning patterns in large student populations

## Getting Help

- Check the example scripts in `.dev/` for working code
- Review the API documentation for detailed parameter descriptions
- Explore the generated CSV files to understand the data structure

## Dependencies

The simulator requires:
- `pydantic` for data validation
- `networkx` for skill dependency graphs
- `matplotlib` for visualization (optional)
- `pydot` for graph layout (optional)

Install with: `pip install pydantic networkx matplotlib pydot`
