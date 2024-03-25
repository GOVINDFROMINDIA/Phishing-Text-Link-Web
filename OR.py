def OR_Gate(x1, y1):
    if x1 == 1 or y1 == 1:
        return 1
    else:
        return 0

def main():
    print("Enter two numbers (either 1 or 0) separated by a space:")
    x_input, y_input = map(int, input().split())

    if x_input not in [0, 1] or y_input not in [0, 1]:
        print("Invalid input. Please enter either 0 or 1.")
        return

    result = OR_Gate(x_input, y_input)
    print("OR Gate output:", result)

if __name__ == "__main__":
    main()
