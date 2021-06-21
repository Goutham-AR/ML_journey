// TODO: use classes to refactor a lot operations on vectors


#include <iostream>
#include <fstream>
#include <string>
#include <charconv>
#include <utility>
#include <vector>
#include <exception>


using vecf = std::vector<float>;
vecf operator*(const vecf& v1, const vecf& v2);
vecf operator*(const vecf& v, float s);
vecf operator*(float s, const vecf& v);
vecf operator-(const vecf& v1, const vecf& v2);
float v_avg(const vecf& v);



std::pair<vecf, vecf> read_from_file(const std::string& filename);
vecf predict(const vecf& X, float b, float w);
std::pair<float, float> train(const vecf& X, const vecf& Y, int iterations, float lr);
std::pair<float, float> gradient(const vecf& X, const vecf& Y, float w, float b);


int main()
{
    auto pair = read_from_file("data.txt");

    // for (int i = 0; i < pair.first.size(); i++)
    // {
    //     std::cout << pair.first[i] << "\t" << pair.second[i] << "\n";
    // }

    auto parameters = train(pair.first, pair.second, 20000, 0.001);
    std::cout << parameters.first << " " << parameters.second << "\n";
}

std::pair<vecf, vecf> read_from_file(const std::string& filename)
{
    vecf keys, values;
    std::ifstream file{ filename, std::ios_base::in };
    std::uint32_t count = 1;
    while (file)
    {
        std::string c;
        file >> c;
        float data;
        auto [p, ec] = std::from_chars(c.data(), c.data() + c.size(), data);
        if (ec == std::errc{})
        {
            (count % 2 != 0) ? keys.push_back(data) : values.push_back(data);
            count++;
        }
    }
    file.close();
    return std::pair{ keys, values };
}

vecf predict(const vecf& X, float b, float w)
{
    vecf predicted_values(X.size());
    for (int i = 0; i < X.size(); i++)
    {
        predicted_values[i] = X[i] * w + b;
    }

    return predicted_values;
}



float v_avg(const vecf& v)
{
    auto sum = 0.0f;
    for (int i = 0; i < v.size(); i++)
        sum += v[i];

    return sum / v.size();
}

// element wise product
vecf operator*(const vecf& v1, const vecf& v2)
{

    if (v1.size() == v2.size())
    {
        vecf product_vector(v1.size());
        for (int i = 0; i < v1.size(); i++)
        {
            product_vector[i] = v1[i] * v2[i];
        }
        return product_vector;
    }
    else
    {
        throw std::exception{ "sizes of the vectors should be same for element wise product" };
    }
}


vecf operator-(const vecf& v1, const vecf& v2)
{

    if (v1.size() == v2.size())
    {
        vecf resultant(v1.size());
        for (int i = 0; i < v1.size(); i++)
        {
            resultant[i] = v1[i] - v2[i];
        }
        return resultant;
    }
    else
    {
        throw std::exception{ "sizes of the vectors should be same for element wise subtraction" };
    }
}

vecf operator*(const vecf& v, float s)
{
    vecf resultant(v.size());
    for (int i = 0; i < v.size(); i++)
    {
        resultant[i] = v[i] * s;
    }
    return resultant;
}

vecf operator*(float s, const vecf& v)
{
    return v * s;
}



std::pair<float, float> gradient(const vecf& X, const vecf& Y, float w, float b)
{
    auto predicted_values = predict(X, b, w);
    try
    {
        auto m_gradient = 2 * v_avg((predicted_values - Y) * X);
        auto b_gradient = 2 * v_avg(predicted_values - Y);

        return std::pair{ m_gradient, b_gradient };
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        std::exit(1);
    }
}

std::pair<float, float> train(const vecf& X, const vecf& Y, int iterations, float lr)
{
    float w = 0, b = 0;
    for (int i = 0; i < iterations; i++)
    {
        auto [w_grad, b_grad] = gradient(X, Y, w, b);
        w -= w_grad * lr;
        b -= b_grad * lr;
    }
    return std::pair{ w, b };
}