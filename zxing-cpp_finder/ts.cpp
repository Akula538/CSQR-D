#include <charconv>
#include <iostream>

int main() {
    int x;
    auto str = "123";
    std::from_chars(str, str+3, x);
    std::cout << x;
}
