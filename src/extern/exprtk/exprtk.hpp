/*
 ******************************************************************
 *           C++ Mathematical Expression Toolkit Library          *
 *                                                                *
 * Author: Arash Partow (1999-2022)                               *
 * URL: https://www.partow.net/programming/exprtk/index.html      *
 *                                                                *
 * Copyright notice:                                              *
 * Free use of the C++ Mathematical Expression Toolkit Library is *
 * permitted under the guidelines and in accordance with the most *
 * current version of the MIT License.                            *
 * https://www.opensource.org/licenses/MIT                        *
 *                                                                *
 * Example expressions:                                           *
 * (00) (y + x / y) * (x - y / x)                                 *
 * (01) (x^2 / sin(2 * pi / y)) - x / 2                           *
 * (02) sqrt(1 - (x^2))                                           *
 * (03) 1 - sin(2 * x) + cos(pi / y)                              *
 * (04) a * exp(2 * t) + c                                        *
 * (05) if(((x + 2) == 3) and ((y + 5) <= 9),1 + w, 2 / z)        *
 * (06) (avg(x,y) <= x + y ? x - y : x * y) + 2 * pi / x          *
 * (07) z := x + sin(2 * pi / y)                                  *
 * (08) u := 2 * (pi * z) / (w := x + cos(y / pi))                *
 * (09) clamp(-1,sin(2 * pi * x) + cos(y / 2 * pi),+1)            *
 * (10) inrange(-2,m,+2) == if(({-2 <= m} and [m <= +2]),1,0)     *
 * (11) (2sin(x)cos(2y)7 + 1) == (2 * sin(x) * cos(2*y) * 7 + 1)  *
 * (12) (x ilike 's*ri?g') and [y < (3 z^7 + w)]                  *
 *                                                                *
 ******************************************************************
*/


#ifndef INCLUDE_EXPRTK_HPP
#define INCLUDE_EXPRTK_HPP


#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <iterator>
#include <limits>
#include <list>
#include <map>
#include <set>
#include <stack>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


namespace exprtk
{
   #ifdef exprtk_enable_debugging
     #define exprtk_debug(params) printf params
   #else
     #define exprtk_debug(params) (void)0
   #endif

   #define exprtk_error_location             \
   "exprtk.hpp:" + details::to_str(__LINE__) \

   #if defined(__GNUC__) && (__GNUC__  >= 7)

      #define exprtk_disable_fallthrough_begin                      \
      _Pragma ("GCC diagnostic push")                               \
      _Pragma ("GCC diagnostic ignored \"-Wimplicit-fallthrough\"") \

      #define exprtk_disable_fallthrough_end                        \
      _Pragma ("GCC diagnostic pop")                                \

   #else
      #define exprtk_disable_fallthrough_begin (void)0;
      #define exprtk_disable_fallthrough_end   (void)0;
   #endif

   #if __cplusplus >= 201103L
      #define exprtk_override override
      #define exprtk_final    final
      #define exprtk_delete   = delete
   #else
      #define exprtk_override
      #define exprtk_final
      #define exprtk_delete
   #endif

   namespace details
   {
      typedef char                   char_t;
      typedef char_t*                char_ptr;
      typedef char_t const*          char_cptr;
      typedef unsigned char          uchar_t;
      typedef uchar_t*               uchar_ptr;
      typedef uchar_t const*         uchar_cptr;
      typedef unsigned long long int _uint64_t;
      typedef long long int          _int64_t;

      inline bool is_whitespace(const char_t c)
      {
         return (' '  == c) || ('\n' == c) ||
                ('\r' == c) || ('\t' == c) ||
                ('\b' == c) || ('\v' == c) ||
                ('\f' == c) ;
      }

      inline bool is_operator_char(const char_t c)
      {
         return ('+' == c) || ('-' == c) ||
                ('*' == c) || ('/' == c) ||
                ('^' == c) || ('<' == c) ||
                ('>' == c) || ('=' == c) ||
                (',' == c) || ('!' == c) ||
                ('(' == c) || (')' == c) ||
                ('[' == c) || (']' == c) ||
                ('{' == c) || ('}' == c) ||
                ('%' == c) || (':' == c) ||
                ('?' == c) || ('&' == c) ||
                ('|' == c) || (';' == c) ;
      }

      inline bool is_letter(const char_t c)
      {
         return (('a' <= c) && (c <= 'z')) ||
                (('A' <= c) && (c <= 'Z')) ;
      }

      inline bool is_digit(const char_t c)
      {
         return ('0' <= c) && (c <= '9');
      }

      inline bool is_letter_or_digit(const char_t c)
      {
         return is_letter(c) || is_digit(c);
      }

      inline bool is_left_bracket(const char_t c)
      {
         return ('(' == c) || ('[' == c) || ('{' == c);
      }

      inline bool is_right_bracket(const char_t c)
      {
         return (')' == c) || (']' == c) || ('}' == c);
      }

      inline bool is_bracket(const char_t c)
      {
         return is_left_bracket(c) || is_right_bracket(c);
      }

      inline bool is_sign(const char_t c)
      {
         return ('+' == c) || ('-' == c);
      }

      inline bool is_invalid(const char_t c)
      {
         return !is_whitespace   (c) &&
                !is_operator_char(c) &&
                !is_letter       (c) &&
                !is_digit        (c) &&
                ('.'  != c)          &&
                ('_'  != c)          &&
                ('$'  != c)          &&
                ('~'  != c)          &&
                ('\'' != c);
      }

      inline bool is_valid_string_char(const char_t c)
      {
         return std::isprint(static_cast<uchar_t>(c)) ||
                is_whitespace(c);
      }

      inline void case_normalise(std::string&)
      {}

      inline bool imatch(const char_t c1, const char_t c2)
      {
         return c1 == c2;
      }

      inline bool imatch(const std::string& s1, const std::string& s2)
      {
         return s1 == s2;
      }

      struct ilesscompare
      {
         inline bool operator() (const std::string& s1, const std::string& s2) const
         {
            return s1 < s2;
         }
      };

      inline bool is_valid_sf_symbol(const std::string& symbol)
      {
         // Special function: $f12 or $F34
         return (4 == symbol.size())  &&
                ('$' == symbol[0])    &&
                imatch('f',symbol[1]) &&
                is_digit(symbol[2])   &&
                is_digit(symbol[3]);
      }

      inline const char_t& front(const std::string& s)
      {
         return s[0];
      }

      inline const char_t& back(const std::string& s)
      {
         return s[s.size() - 1];
      }

      inline std::string to_str(int i)
      {
         if (0 == i)
            return std::string("0");

         std::string result;

         const int sign = (i < 0) ? -1 : 1;

         for ( ; i; i /= 10)
         {
            result += '0' + static_cast<char_t>(sign * (i % 10));
         }

         if (sign < 0)
         {
            result += '-';
         }

         std::reverse(result.begin(), result.end());


         return result;
      }

      inline std::string to_str(std::size_t i)
      {
         return to_str(static_cast<int>(i));
      }

      inline bool is_hex_digit(const uchar_t digit)
      {
         return (('0' <= digit) && (digit <= '9')) ||
                (('A' <= digit) && (digit <= 'F')) ||
                (('a' <= digit) && (digit <= 'f')) ;
      }

      inline uchar_t hex_to_bin(uchar_t h)
      {
         if (('0' <= h) && (h <= '9'))
            return (h - '0');
         else
            return static_cast<uchar_t>(std::toupper(h) - 'A');
      }

      template <typename Iterator>
      inline bool parse_hex(Iterator& itr, Iterator end,
                            char_t& result)
      {
         if (
              (end ==  (itr    ))               ||
              (end ==  (itr + 1))               ||
              (end ==  (itr + 2))               ||
              (end ==  (itr + 3))               ||
              ('0' != *(itr    ))               ||
              ('X' != std::toupper(*(itr + 1))) ||
              (!is_hex_digit(*(itr + 2)))       ||
              (!is_hex_digit(*(itr + 3)))
            )
         {
            return false;
         }

         result = hex_to_bin(static_cast<uchar_t>(*(itr + 2))) << 4 |
                  hex_to_bin(static_cast<uchar_t>(*(itr + 3))) ;

         return true;
      }

      inline bool cleanup_escapes(std::string& s)
      {
         typedef std::string::iterator str_itr_t;

         str_itr_t itr1 = s.begin();
         str_itr_t itr2 = s.begin();
         str_itr_t end  = s.end  ();

         std::size_t removal_count  = 0;

         while (end != itr1)
         {
            if ('\\' == (*itr1))
            {
               if (end == ++itr1)
               {
                  return false;
               }
               else if (parse_hex(itr1, end, *itr2))
               {
                  itr1+= 4;
                  itr2+= 1;
                  removal_count +=4;
               }
               else if ('a' == (*itr1)) { (*itr2++) = '\a'; ++itr1; ++removal_count; }
               else if ('b' == (*itr1)) { (*itr2++) = '\b'; ++itr1; ++removal_count; }
               else if ('f' == (*itr1)) { (*itr2++) = '\f'; ++itr1; ++removal_count; }
               else if ('n' == (*itr1)) { (*itr2++) = '\n'; ++itr1; ++removal_count; }
               else if ('r' == (*itr1)) { (*itr2++) = '\r'; ++itr1; ++removal_count; }
               else if ('t' == (*itr1)) { (*itr2++) = '\t'; ++itr1; ++removal_count; }
               else if ('v' == (*itr1)) { (*itr2++) = '\v'; ++itr1; ++removal_count; }
               else if ('0' == (*itr1)) { (*itr2++) = '\0'; ++itr1; ++removal_count; }
               else
               {
                  (*itr2++) = (*itr1++);
                  ++removal_count;
               }
               continue;
            }
            else
               (*itr2++) = (*itr1++);
         }

         if ((removal_count > s.size()) || (0 == removal_count))
            return false;

         s.resize(s.size() - removal_count);

         return true;
      }

      class build_string
      {
      public:

         build_string(const std::size_t& initial_size = 64)
         {
            data_.reserve(initial_size);
         }

         inline build_string& operator << (const std::string& s)
         {
            data_ += s;
            return (*this);
         }

         inline build_string& operator << (char_cptr s)
         {
            data_ += std::string(s);
            return (*this);
         }

         inline operator std::string () const
         {
            return data_;
         }

         inline std::string as_string() const
         {
            return data_;
         }

      private:

         std::string data_;
      };

      static const std::string reserved_words[] =
                                  {
                                    "break",  "case",  "continue",  "default",  "false",  "for",
                                    "if", "else", "ilike",  "in", "like", "and",  "nand", "nor",
                                    "not",  "null",  "or",   "repeat", "return",  "shl",  "shr",
                                    "swap", "switch", "true",  "until", "var",  "while", "xnor",
                                    "xor", "&", "|"
                                  };

      static const std::size_t reserved_words_size = sizeof(reserved_words) / sizeof(std::string);

      static const std::string reserved_symbols[] =
                                  {
                                    "abs",  "acos",  "acosh",  "and",  "asin",  "asinh", "atan",
                                    "atanh", "atan2", "avg",  "break", "case", "ceil",  "clamp",
                                    "continue",   "cos",   "cosh",   "cot",   "csc",  "default",
                                    "deg2grad",  "deg2rad",   "equal",  "erf",   "erfc",  "exp",
                                    "expm1",  "false",   "floor",  "for",   "frac",  "grad2deg",
                                    "hypot", "iclamp", "if",  "else", "ilike", "in",  "inrange",
                                    "like",  "log",  "log10", "log2",  "logn",  "log1p", "mand",
                                    "max", "min",  "mod", "mor",  "mul", "ncdf",  "nand", "nor",
                                    "not",   "not_equal",   "null",   "or",   "pow",  "rad2deg",
                                    "repeat", "return", "root", "round", "roundn", "sec", "sgn",
                                    "shl", "shr", "sin", "sinc", "sinh", "sqrt",  "sum", "swap",
                                    "switch", "tan",  "tanh", "true",  "trunc", "until",  "var",
                                    "while", "xnor", "xor", "&", "|"
                                  };

      static const std::size_t reserved_symbols_size = sizeof(reserved_symbols) / sizeof(std::string);

      static const std::string base_function_list[] =
                                  {
                                    "abs", "acos",  "acosh", "asin",  "asinh", "atan",  "atanh",
                                    "atan2",  "avg",  "ceil",  "clamp",  "cos",  "cosh",  "cot",
                                    "csc",  "equal",  "erf",  "erfc",  "exp",  "expm1", "floor",
                                    "frac", "hypot", "iclamp",  "like", "log", "log10",  "log2",
                                    "logn", "log1p", "mand", "max", "min", "mod", "mor",  "mul",
                                    "ncdf",  "pow",  "root",  "round",  "roundn",  "sec", "sgn",
                                    "sin", "sinc", "sinh", "sqrt", "sum", "swap", "tan", "tanh",
                                    "trunc",  "not_equal",  "inrange",  "deg2grad",   "deg2rad",
                                    "rad2deg", "grad2deg"
                                  };

      static const std::size_t base_function_list_size = sizeof(base_function_list) / sizeof(std::string);

      static const std::string logic_ops_list[] =
                                  {
                                    "and", "nand", "nor", "not", "or",  "xnor", "xor", "&", "|"
                                  };

      static const std::size_t logic_ops_list_size = sizeof(logic_ops_list) / sizeof(std::string);

      static const std::string cntrl_struct_list[] =
                                  {
                                     "if", "switch", "for", "while", "repeat", "return"
                                  };

      static const std::size_t cntrl_struct_list_size = sizeof(cntrl_struct_list) / sizeof(std::string);

      static const std::string arithmetic_ops_list[] =
                                  {
                                    "+", "-", "*", "/", "%", "^"
                                  };

      static const std::size_t arithmetic_ops_list_size = sizeof(arithmetic_ops_list) / sizeof(std::string);

      static const std::string assignment_ops_list[] =
                                  {
                                    ":=", "+=", "-=",
                                    "*=", "/=", "%="
                                  };

      static const std::size_t assignment_ops_list_size = sizeof(assignment_ops_list) / sizeof(std::string);

      static const std::string inequality_ops_list[] =
                                  {
                                     "<",  "<=", "==",
                                     "=",  "!=", "<>",
                                    ">=",  ">"
                                  };

      static const std::size_t inequality_ops_list_size = sizeof(inequality_ops_list) / sizeof(std::string);

      inline bool is_reserved_word(const std::string& symbol)
      {
         for (std::size_t i = 0; i < reserved_words_size; ++i)
         {
            if (imatch(symbol, reserved_words[i]))
            {
               return true;
            }
         }

         return false;
      }

      inline bool is_reserved_symbol(const std::string& symbol)
      {
         for (std::size_t i = 0; i < reserved_symbols_size; ++i)
         {
            if (imatch(symbol, reserved_symbols[i]))
            {
               return true;
            }
         }

         return false;
      }

      inline bool is_base_function(const std::string& function_name)
      {
         for (std::size_t i = 0; i < base_function_list_size; ++i)
         {
            if (imatch(function_name, base_function_list[i]))
            {
               return true;
            }
         }

         return false;
      }

      inline bool is_control_struct(const std::string& cntrl_strct)
      {
         for (std::size_t i = 0; i < cntrl_struct_list_size; ++i)
         {
            if (imatch(cntrl_strct, cntrl_struct_list[i]))
            {
               return true;
            }
         }

         return false;
      }

      inline bool is_logic_opr(const std::string& lgc_opr)
      {
         for (std::size_t i = 0; i < logic_ops_list_size; ++i)
         {
            if (imatch(lgc_opr, logic_ops_list[i]))
            {
               return true;
            }
         }

         return false;
      }

      struct cs_match
      {
         static inline bool cmp(const char_t c0, const char_t c1)
         {
            return (c0 == c1);
         }
      };

      struct cis_match
      {
         static inline bool cmp(const char_t c0, const char_t c1)
         {
            return (std::tolower(c0) == std::tolower(c1));
         }
      };

      template <typename Iterator, typename Compare>
      inline bool match_impl(const Iterator pattern_begin,
                             const Iterator pattern_end  ,
                             const Iterator data_begin   ,
                             const Iterator data_end     ,
                             const typename std::iterator_traits<Iterator>::value_type& zero_or_more,
                             const typename std::iterator_traits<Iterator>::value_type& exactly_one )
      {
         const Iterator null_itr(0);

         Iterator p_itr  = pattern_begin;
         Iterator d_itr  = data_begin;
         Iterator np_itr = null_itr;
         Iterator nd_itr = null_itr;

         for ( ; ; )
         {
            const bool pvalid = p_itr != pattern_end;
            const bool dvalid = d_itr != data_end;

            if (!pvalid && !dvalid)
               break;

            if (pvalid)
            {
               const typename std::iterator_traits<Iterator>::value_type c = *(p_itr);

               if (zero_or_more == c)
               {
                  np_itr = p_itr;
                  nd_itr = d_itr + 1;
                  ++p_itr;
                  continue;
               }
               else if (dvalid && ((exactly_one == c) || Compare::cmp(c,*(d_itr))))
               {
                  ++p_itr;
                  ++d_itr;
                  continue;
               }
            }

            if ((null_itr != nd_itr) && (nd_itr <= data_end))
            {
               p_itr = np_itr;
               d_itr = nd_itr;
               continue;
            }

            return false;
         }

         return true;
      }

      inline bool wc_match(const std::string& wild_card,
                           const std::string& str)
      {
         return match_impl<char_cptr,cs_match>(
                   wild_card.data(),
                   wild_card.data() + wild_card.size(),
                   str.data(),
                   str.data() + str.size(),
                   '*', '?');
      }

      inline bool wc_imatch(const std::string& wild_card,
                            const std::string& str)
      {
         return match_impl<char_cptr,cis_match>(
                   wild_card.data(),
                   wild_card.data() + wild_card.size(),
                   str.data(),
                   str.data() + str.size(),
                   '*', '?');
      }

      inline bool sequence_match(const std::string& pattern,
                                 const std::string& str,
                                 std::size_t&       diff_index,
                                 char_t&            diff_value)
      {
         if (str.empty())
         {
            return ("Z" == pattern);
         }
         else if ('*' == pattern[0])
            return false;

         typedef std::string::const_iterator itr_t;

         itr_t p_itr = pattern.begin();
         itr_t s_itr = str    .begin();

         const itr_t p_end = pattern.end();
         const itr_t s_end = str    .end();

         while ((s_end != s_itr) && (p_end != p_itr))
         {
            if ('*' == (*p_itr))
            {
               const char_t target = static_cast<char_t>(std::toupper(*(p_itr - 1)));

               if ('*' == target)
               {
                  diff_index = static_cast<std::size_t>(std::distance(str.begin(),s_itr));
                  diff_value = static_cast<char_t>(std::toupper(*p_itr));

                  return false;
               }
               else
                  ++p_itr;

               while (s_itr != s_end)
               {
                  if (target != std::toupper(*s_itr))
                     break;
                  else
                     ++s_itr;
               }

               continue;
            }
            else if (
                      ('?' != *p_itr) &&
                      std::toupper(*p_itr) != std::toupper(*s_itr)
                    )
            {
               diff_index = static_cast<std::size_t>(std::distance(str.begin(),s_itr));
               diff_value = static_cast<char_t>(std::toupper(*p_itr));

               return false;
            }

            ++p_itr;
            ++s_itr;
         }

         return (
                  (s_end == s_itr) &&
                  (
                    (p_end ==  p_itr) ||
                    ('*'   == *p_itr)
                  )
                );
      }

      static const double pow10[] = {
                                      1.0,
                                      1.0E+001, 1.0E+002, 1.0E+003, 1.0E+004,
                                      1.0E+005, 1.0E+006, 1.0E+007, 1.0E+008,
                                      1.0E+009, 1.0E+010, 1.0E+011, 1.0E+012,
                                      1.0E+013, 1.0E+014, 1.0E+015, 1.0E+016
                                    };

      static const std::size_t pow10_size = sizeof(pow10) / sizeof(double);

      namespace numeric
      {
         namespace constant
         {
            static const double e       =  2.71828182845904523536028747135266249775724709369996;
            static const double pi      =  3.14159265358979323846264338327950288419716939937510;
            static const double pi_2    =  1.57079632679489661923132169163975144209858469968755;
            static const double pi_4    =  0.78539816339744830961566084581987572104929234984378;
            static const double pi_180  =  0.01745329251994329576923690768488612713442871888542;
            static const double _1_pi   =  0.31830988618379067153776752674502872406891929148091;
            static const double _2_pi   =  0.63661977236758134307553505349005744813783858296183;
            static const double _180_pi = 57.29577951308232087679815481410517033240547246656443;
            static const double log2    =  0.69314718055994530941723212145817656807550013436026;
            static const double sqrt2   =  1.41421356237309504880168872420969807856967187537695;
         }

         namespace details
         {
            struct unknown_type_tag { unknown_type_tag() {} };
            struct real_type_tag    { real_type_tag   () {} };
            struct complex_type_tag { complex_type_tag() {} };
            struct int_type_tag     { int_type_tag    () {} };

            template <typename T>
            struct number_type
            {
               typedef unknown_type_tag type;
               number_type() {}
            };

            #define exprtk_register_real_type_tag(T)             \
            template <> struct number_type<T>                    \
            { typedef real_type_tag type; number_type() {} };    \

            #define exprtk_register_complex_type_tag(T)          \
            template <> struct number_type<std::complex<T> >     \
            { typedef complex_type_tag type; number_type() {} }; \

            #define exprtk_register_int_type_tag(T)              \
            template <> struct number_type<T>                    \
            { typedef int_type_tag type; number_type() {} };     \

            exprtk_register_real_type_tag(double     )
            exprtk_register_real_type_tag(long double)
            exprtk_register_real_type_tag(float      )

            exprtk_register_complex_type_tag(double     )
            exprtk_register_complex_type_tag(long double)
            exprtk_register_complex_type_tag(float      )

            exprtk_register_int_type_tag(short         )
            exprtk_register_int_type_tag(int           )
            exprtk_register_int_type_tag(_int64_t      )
            exprtk_register_int_type_tag(unsigned short)
            exprtk_register_int_type_tag(unsigned int  )
            exprtk_register_int_type_tag(_uint64_t     )

            #undef exprtk_register_real_type_tag
            #undef exprtk_register_int_type_tag

            template <typename T>
            struct epsilon_type {};

            #define exprtk_define_epsilon_type(Type, Epsilon)      \
            template <> struct epsilon_type<Type>                  \
            {                                                      \
               static inline Type value()                          \
               {                                                   \
                  const Type epsilon = static_cast<Type>(Epsilon); \
                  return epsilon;                                  \
               }                                                   \
            };                                                     \

            exprtk_define_epsilon_type(float      , 0.00000100000f)
            exprtk_define_epsilon_type(double     , 0.000000000100)
            exprtk_define_epsilon_type(long double, 0.000000000001)

            #undef exprtk_define_epsilon_type

            template <typename T>
            inline bool is_nan_impl(const T v, real_type_tag)
            {
               return std::not_equal_to<T>()(v,v);
            }

            template <typename T>
            inline int to_int32_impl(const T v, real_type_tag)
            {
               return static_cast<int>(v);
            }

            template <typename T>
            inline _int64_t to_int64_impl(const T v, real_type_tag)
            {
               return static_cast<_int64_t>(v);
            }

            template <typename T>
            inline bool is_true_impl(const T v)
            {
               return std::not_equal_to<T>()(T(0),v);
            }

            template <typename T>
            inline bool is_false_impl(const T v)
            {
               return std::equal_to<T>()(T(0),v);
            }

            template <typename T>
            inline T abs_impl(const T v, real_type_tag)
            {
               return ((v < T(0)) ? -v : v);
            }

            template <typename T>
            inline T min_impl(const T v0, const T v1, real_type_tag)
            {
               return std::min<T>(v0,v1);
            }

            template <typename T>
            inline T max_impl(const T v0, const T v1, real_type_tag)
            {
               return std::max<T>(v0,v1);
            }

            template <typename T>
            inline T equal_impl(const T v0, const T v1, real_type_tag)
            {
               const T epsilon = epsilon_type<T>::value();
               return (abs_impl(v0 - v1,real_type_tag()) <= (std::max(T(1),std::max(abs_impl(v0,real_type_tag()),abs_impl(v1,real_type_tag()))) * epsilon)) ? T(1) : T(0);
            }

            inline float equal_impl(const float v0, const float v1, real_type_tag)
            {
               const float epsilon = epsilon_type<float>::value();
               return (abs_impl(v0 - v1,real_type_tag()) <= (std::max(1.0f,std::max(abs_impl(v0,real_type_tag()),abs_impl(v1,real_type_tag()))) * epsilon)) ? 1.0f : 0.0f;
            }

            template <typename T>
            inline T equal_impl(const T v0, const T v1, int_type_tag)
            {
               return (v0 == v1) ? 1 : 0;
            }

            template <typename T>
            inline T expm1_impl(const T v, real_type_tag)
            {
               // return std::expm1<T>(v);
               if (abs_impl(v,real_type_tag()) < T(0.00001))
                  return v + (T(0.5) * v * v);
               else
                  return std::exp(v) - T(1);
            }

            template <typename T>
            inline T expm1_impl(const T v, int_type_tag)
            {
               return T(std::exp<double>(v)) - T(1);
            }

            template <typename T>
            inline T nequal_impl(const T v0, const T v1, real_type_tag)
            {
               typedef real_type_tag rtg;
               const T epsilon = epsilon_type<T>::value();
               return (abs_impl(v0 - v1,rtg()) > (std::max(T(1),std::max(abs_impl(v0,rtg()),abs_impl(v1,rtg()))) * epsilon)) ? T(1) : T(0);
            }

            inline float nequal_impl(const float v0, const float v1, real_type_tag)
            {
               typedef real_type_tag rtg;
               const float epsilon = epsilon_type<float>::value();
               return (abs_impl(v0 - v1,rtg()) > (std::max(1.0f,std::max(abs_impl(v0,rtg()),abs_impl(v1,rtg()))) * epsilon)) ? 1.0f : 0.0f;
            }

            template <typename T>
            inline T nequal_impl(const T v0, const T v1, int_type_tag)
            {
               return (v0 != v1) ? 1 : 0;
            }

            template <typename T>
            inline T modulus_impl(const T v0, const T v1, real_type_tag)
            {
               return std::fmod(v0,v1);
            }

            template <typename T>
            inline T modulus_impl(const T v0, const T v1, int_type_tag)
            {
               return v0 % v1;
            }

            template <typename T>
            inline T pow_impl(const T v0, const T v1, real_type_tag)
            {
               return std::pow(v0,v1);
            }

            template <typename T>
            inline T pow_impl(const T v0, const T v1, int_type_tag)
            {
               return std::pow(static_cast<double>(v0),static_cast<double>(v1));
            }

            template <typename T>
            inline T logn_impl(const T v0, const T v1, real_type_tag)
            {
               return std::log(v0) / std::log(v1);
            }

            template <typename T>
            inline T logn_impl(const T v0, const T v1, int_type_tag)
            {
               return static_cast<T>(logn_impl<double>(static_cast<double>(v0),static_cast<double>(v1),real_type_tag()));
            }

            template <typename T>
            inline T log1p_impl(const T v, real_type_tag)
            {
               if (v > T(-1))
               {
                  if (abs_impl(v,real_type_tag()) > T(0.0001))
                  {
                     return std::log(T(1) + v);
                  }
                  else
                     return (T(-0.5) * v + T(1)) * v;
               }
               else
                  return std::numeric_limits<T>::quiet_NaN();
            }

            template <typename T>
            inline T log1p_impl(const T v, int_type_tag)
            {
               if (v > T(-1))
               {
                  return std::log(T(1) + v);
               }
               else
                  return std::numeric_limits<T>::quiet_NaN();
            }

            template <typename T>
            inline T root_impl(const T v0, const T v1, real_type_tag)
            {
               if (v1 < T(0))
                  return std::numeric_limits<T>::quiet_NaN();

               const std::size_t n = static_cast<std::size_t>(v1);

               if ((v0 < T(0)) && (0 == (n % 2)))
                  return std::numeric_limits<T>::quiet_NaN();

               return std::pow(v0, T(1) / n);
            }

            template <typename T>
            inline T root_impl(const T v0, const T v1, int_type_tag)
            {
               return root_impl<double>(static_cast<double>(v0),static_cast<double>(v1),real_type_tag());
            }

            template <typename T>
            inline T round_impl(const T v, real_type_tag)
            {
               return ((v < T(0)) ? std::ceil(v - T(0.5)) : std::floor(v + T(0.5)));
            }

            template <typename T>
            inline T roundn_impl(const T v0, const T v1, real_type_tag)
            {
               const int index = std::max<int>(0, std::min<int>(pow10_size - 1, static_cast<int>(std::floor(v1))));
               const T p10 = T(pow10[index]);

               if (v0 < T(0))
                  return T(std::ceil ((v0 * p10) - T(0.5)) / p10);
               else
                  return T(std::floor((v0 * p10) + T(0.5)) / p10);
            }

            template <typename T>
            inline T roundn_impl(const T v0, const T, int_type_tag)
            {
               return v0;
            }

            template <typename T>
            inline T hypot_impl(const T v0, const T v1, real_type_tag)
            {
               return std::sqrt((v0 * v0) + (v1 * v1));
            }

            template <typename T>
            inline T hypot_impl(const T v0, const T v1, int_type_tag)
            {
               return static_cast<T>(std::sqrt(static_cast<double>((v0 * v0) + (v1 * v1))));
            }

            template <typename T>
            inline T atan2_impl(const T v0, const T v1, real_type_tag)
            {
               return std::atan2(v0,v1);
            }

            template <typename T>
            inline T atan2_impl(const T, const T, int_type_tag)
            {
               return 0;
            }

            template <typename T>
            inline T shr_impl(const T v0, const T v1, real_type_tag)
            {
               return v0 * (T(1) / std::pow(T(2),static_cast<T>(static_cast<int>(v1))));
            }

            template <typename T>
            inline T shr_impl(const T v0, const T v1, int_type_tag)
            {
               return v0 >> v1;
            }

            template <typename T>
            inline T shl_impl(const T v0, const T v1, real_type_tag)
            {
               return v0 * std::pow(T(2),static_cast<T>(static_cast<int>(v1)));
            }

            template <typename T>
            inline T shl_impl(const T v0, const T v1, int_type_tag)
            {
               return v0 << v1;
            }

            template <typename T>
            inline T sgn_impl(const T v, real_type_tag)
            {
               if      (v > T(0)) return T(+1);
               else if (v < T(0)) return T(-1);
               else               return T( 0);
            }

            template <typename T>
            inline T sgn_impl(const T v, int_type_tag)
            {
               if      (v > T(0)) return T(+1);
               else if (v < T(0)) return T(-1);
               else               return T( 0);
            }

            template <typename T>
            inline T and_impl(const T v0, const T v1, real_type_tag)
            {
               return (is_true_impl(v0) && is_true_impl(v1)) ? T(1) : T(0);
            }

            template <typename T>
            inline T and_impl(const T v0, const T v1, int_type_tag)
            {
               return v0 && v1;
            }

            template <typename T>
            inline T nand_impl(const T v0, const T v1, real_type_tag)
            {
               return (is_false_impl(v0) || is_false_impl(v1)) ? T(1) : T(0);
            }

            template <typename T>
            inline T nand_impl(const T v0, const T v1, int_type_tag)
            {
               return !(v0 && v1);
            }

            template <typename T>
            inline T or_impl(const T v0, const T v1, real_type_tag)
            {
               return (is_true_impl(v0) || is_true_impl(v1)) ? T(1) : T(0);
            }

            template <typename T>
            inline T or_impl(const T v0, const T v1, int_type_tag)
            {
               return (v0 || v1);
            }

            template <typename T>
            inline T nor_impl(const T v0, const T v1, real_type_tag)
            {
               return (is_false_impl(v0) && is_false_impl(v1)) ? T(1) : T(0);
            }

            template <typename T>
            inline T nor_impl(const T v0, const T v1, int_type_tag)
            {
               return !(v0 || v1);
            }

            template <typename T>
            inline T xor_impl(const T v0, const T v1, real_type_tag)
            {
               return (is_false_impl(v0) != is_false_impl(v1)) ? T(1) : T(0);
            }

            template <typename T>
            inline T xor_impl(const T v0, const T v1, int_type_tag)
            {
               return v0 ^ v1;
            }

            template <typename T>
            inline T xnor_impl(const T v0, const T v1, real_type_tag)
            {
               const bool v0_true = is_true_impl(v0);
               const bool v1_true = is_true_impl(v1);

               if ((v0_true &&  v1_true) || (!v0_true && !v1_true))
                  return T(1);
               else
                  return T(0);
            }

            template <typename T>
            inline T xnor_impl(const T v0, const T v1, int_type_tag)
            {
               const bool v0_true = is_true_impl(v0);
               const bool v1_true = is_true_impl(v1);

               if ((v0_true &&  v1_true) || (!v0_true && !v1_true))
                  return T(1);
               else
                  return T(0);
            }

            #if (defined(_MSC_VER) && (_MSC_VER >= 1900)) || !defined(_MSC_VER)
            #define exprtk_define_erf(TT, impl)                \
            inline TT erf_impl(const TT v) { return impl(v); } \

            exprtk_define_erf(      float,::erff)
            exprtk_define_erf(     double,::erf )
            exprtk_define_erf(long double,::erfl)
            #undef exprtk_define_erf
            #endif

            template <typename T>
            inline T erf_impl(const T v, real_type_tag)
            {
               #if defined(_MSC_VER) && (_MSC_VER < 1900)
               // Credits: Abramowitz & Stegun Equations 7.1.25-28
               static const T c[] = {
                                      T( 1.26551223), T(1.00002368),
                                      T( 0.37409196), T(0.09678418),
                                      T(-0.18628806), T(0.27886807),
                                      T(-1.13520398), T(1.48851587),
                                      T(-0.82215223), T(0.17087277)
                                    };

               const T t = T(1) / (T(1) + T(0.5) * abs_impl(v,real_type_tag()));

               const T result = T(1) - t * std::exp((-v * v) -
                                            c[0] + t * (c[1] + t *
                                           (c[2] + t * (c[3] + t *
                                           (c[4] + t * (c[5] + t *
                                           (c[6] + t * (c[7] + t *
                                           (c[8] + t * (c[9]))))))))));

               return (v >= T(0)) ? result : -result;
               #else
               return erf_impl(v);
               #endif
            }

            template <typename T>
            inline T erf_impl(const T v, int_type_tag)
            {
               return erf_impl(static_cast<double>(v),real_type_tag());
            }

            #if (defined(_MSC_VER) && (_MSC_VER >= 1900)) || !defined(_MSC_VER)
            #define exprtk_define_erfc(TT, impl)                \
            inline TT erfc_impl(const TT v) { return impl(v); } \

            exprtk_define_erfc(float      ,::erfcf)
            exprtk_define_erfc(double     ,::erfc )
            exprtk_define_erfc(long double,::erfcl)
            #undef exprtk_define_erfc
            #endif

            template <typename T>
            inline T erfc_impl(const T v, real_type_tag)
            {
               #if defined(_MSC_VER) && (_MSC_VER < 1900)
               return T(1) - erf_impl(v,real_type_tag());
               #else
               return erfc_impl(v);
               #endif
            }

            template <typename T>
            inline T erfc_impl(const T v, int_type_tag)
            {
               return erfc_impl(static_cast<double>(v),real_type_tag());
            }

            template <typename T>
            inline T ncdf_impl(const T v, real_type_tag)
            {
               const T cnd = T(0.5) * (T(1) +
                             erf_impl(abs_impl(v,real_type_tag()) /
                                      T(numeric::constant::sqrt2),real_type_tag()));
               return  (v < T(0)) ? (T(1) - cnd) : cnd;
            }

            template <typename T>
            inline T ncdf_impl(const T v, int_type_tag)
            {
               return ncdf_impl(static_cast<double>(v),real_type_tag());
            }

            template <typename T>
            inline T sinc_impl(const T v, real_type_tag)
            {
               if (std::abs(v) >= std::numeric_limits<T>::epsilon())
                   return(std::sin(v) / v);
               else
                  return T(1);
            }

            template <typename T>
            inline T sinc_impl(const T v, int_type_tag)
            {
               return sinc_impl(static_cast<double>(v),real_type_tag());
            }

            template <typename T> inline T  acos_impl(const T v, real_type_tag) { return std::acos (v); }
            template <typename T> inline T acosh_impl(const T v, real_type_tag) { return std::log(v + std::sqrt((v * v) - T(1))); }
            template <typename T> inline T  asin_impl(const T v, real_type_tag) { return std::asin (v); }
            template <typename T> inline T asinh_impl(const T v, real_type_tag) { return std::log(v + std::sqrt((v * v) + T(1))); }
            template <typename T> inline T  atan_impl(const T v, real_type_tag) { return std::atan (v); }
            template <typename T> inline T atanh_impl(const T v, real_type_tag) { return (std::log(T(1) + v) - std::log(T(1) - v)) / T(2); }
            template <typename T> inline T  ceil_impl(const T v, real_type_tag) { return std::ceil (v); }
            template <typename T> inline T   cos_impl(const T v, real_type_tag) { return std::cos  (v); }
            template <typename T> inline T  cosh_impl(const T v, real_type_tag) { return std::cosh (v); }
            template <typename T> inline T   exp_impl(const T v, real_type_tag) { return std::exp  (v); }
            template <typename T> inline T floor_impl(const T v, real_type_tag) { return std::floor(v); }
            template <typename T> inline T   log_impl(const T v, real_type_tag) { return std::log  (v); }
            template <typename T> inline T log10_impl(const T v, real_type_tag) { return std::log10(v); }
            template <typename T> inline T  log2_impl(const T v, real_type_tag) { return std::log(v)/T(numeric::constant::log2); }
            template <typename T> inline T   neg_impl(const T v, real_type_tag) { return -v;            }
            template <typename T> inline T   pos_impl(const T v, real_type_tag) { return +v;            }
            template <typename T> inline T   sin_impl(const T v, real_type_tag) { return std::sin  (v); }
            template <typename T> inline T  sinh_impl(const T v, real_type_tag) { return std::sinh (v); }
            template <typename T> inline T  sqrt_impl(const T v, real_type_tag) { return std::sqrt (v); }
            template <typename T> inline T   tan_impl(const T v, real_type_tag) { return std::tan  (v); }
            template <typename T> inline T  tanh_impl(const T v, real_type_tag) { return std::tanh (v); }
            template <typename T> inline T   cot_impl(const T v, real_type_tag) { return T(1) / std::tan(v); }
            template <typename T> inline T   sec_impl(const T v, real_type_tag) { return T(1) / std::cos(v); }
            template <typename T> inline T   csc_impl(const T v, real_type_tag) { return T(1) / std::sin(v); }
            template <typename T> inline T   r2d_impl(const T v, real_type_tag) { return (v * T(numeric::constant::_180_pi)); }
            template <typename T> inline T   d2r_impl(const T v, real_type_tag) { return (v * T(numeric::constant::pi_180));  }
            template <typename T> inline T   d2g_impl(const T v, real_type_tag) { return (v * T(10.0/9.0)); }
            template <typename T> inline T   g2d_impl(const T v, real_type_tag) { return (v * T(9.0/10.0)); }
            template <typename T> inline T  notl_impl(const T v, real_type_tag) { return (std::not_equal_to<T>()(T(0),v) ? T(0) : T(1)); }
            template <typename T> inline T  frac_impl(const T v, real_type_tag) { return (v - static_cast<long long>(v)); }
            template <typename T> inline T trunc_impl(const T v, real_type_tag) { return T(static_cast<long long>(v));    }

            template <typename T> inline T   const_pi_impl(real_type_tag) { return T(numeric::constant::pi);            }
            template <typename T> inline T    const_e_impl(real_type_tag) { return T(numeric::constant::e);             }
            template <typename T> inline T const_qnan_impl(real_type_tag) { return std::numeric_limits<T>::quiet_NaN(); }

            template <typename T> inline T   abs_impl(const T v, int_type_tag) { return ((v >= T(0)) ? v : -v); }
            template <typename T> inline T   exp_impl(const T v, int_type_tag) { return std::exp  (v); }
            template <typename T> inline T   log_impl(const T v, int_type_tag) { return std::log  (v); }
            template <typename T> inline T log10_impl(const T v, int_type_tag) { return std::log10(v); }
            template <typename T> inline T  log2_impl(const T v, int_type_tag) { return std::log(v)/T(numeric::constant::log2); }
            template <typename T> inline T   neg_impl(const T v, int_type_tag) { return -v;            }
            template <typename T> inline T   pos_impl(const T v, int_type_tag) { return +v;            }
            template <typename T> inline T  ceil_impl(const T v, int_type_tag) { return v;             }
            template <typename T> inline T floor_impl(const T v, int_type_tag) { return v;             }
            template <typename T> inline T round_impl(const T v, int_type_tag) { return v;             }
            template <typename T> inline T  notl_impl(const T v, int_type_tag) { return !v;            }
            template <typename T> inline T  sqrt_impl(const T v, int_type_tag) { return std::sqrt (v); }
            template <typename T> inline T  frac_impl(const T  , int_type_tag) { return T(0);          }
            template <typename T> inline T trunc_impl(const T v, int_type_tag) { return v;             }
            template <typename T> inline T  acos_impl(const T  , int_type_tag) { return std::numeric_limits<T>::quiet_NaN(); }
            template <typename T> inline T acosh_impl(const T  , int_type_tag) { return std::numeric_limits<T>::quiet_NaN(); }
            template <typename T> inline T  asin_impl(const T  , int_type_tag) { return std::numeric_limits<T>::quiet_NaN(); }
            template <typename T> inline T asinh_impl(const T  , int_type_tag) { return std::numeric_limits<T>::quiet_NaN(); }
            template <typename T> inline T  atan_impl(const T  , int_type_tag) { return std::numeric_limits<T>::quiet_NaN(); }
            template <typename T> inline T atanh_impl(const T  , int_type_tag) { return std::numeric_limits<T>::quiet_NaN(); }
            template <typename T> inline T   cos_impl(const T  , int_type_tag) { return std::numeric_limits<T>::quiet_NaN(); }
            template <typename T> inline T  cosh_impl(const T  , int_type_tag) { return std::numeric_limits<T>::quiet_NaN(); }
            template <typename T> inline T   sin_impl(const T  , int_type_tag) { return std::numeric_limits<T>::quiet_NaN(); }
            template <typename T> inline T  sinh_impl(const T  , int_type_tag) { return std::numeric_limits<T>::quiet_NaN(); }
            template <typename T> inline T   tan_impl(const T  , int_type_tag) { return std::numeric_limits<T>::quiet_NaN(); }
            template <typename T> inline T  tanh_impl(const T  , int_type_tag) { return std::numeric_limits<T>::quiet_NaN(); }
            template <typename T> inline T   cot_impl(const T  , int_type_tag) { return std::numeric_limits<T>::quiet_NaN(); }
            template <typename T> inline T   sec_impl(const T  , int_type_tag) { return std::numeric_limits<T>::quiet_NaN(); }
            template <typename T> inline T   csc_impl(const T  , int_type_tag) { return std::numeric_limits<T>::quiet_NaN(); }

            template <typename T>
            inline bool is_integer_impl(const T& v, real_type_tag)
            {
               return std::equal_to<T>()(T(0),std::fmod(v,T(1)));
            }

            template <typename T>
            inline bool is_integer_impl(const T&, int_type_tag)
            {
               return true;
            }
         }

         template <typename Type>
         struct numeric_info { enum { length = 0, size = 32, bound_length = 0, min_exp = 0, max_exp = 0 }; };

         template <> struct numeric_info<int        > { enum { length = 10, size = 16, bound_length = 9 }; };
         template <> struct numeric_info<float      > { enum { min_exp =  -38, max_exp =  +38 }; };
         template <> struct numeric_info<double     > { enum { min_exp = -308, max_exp = +308 }; };
         template <> struct numeric_info<long double> { enum { min_exp = -308, max_exp = +308 }; };

         template <typename T>
         inline int to_int32(const T v)
         {
            const typename details::number_type<T>::type num_type;
            return to_int32_impl(v, num_type);
         }

         template <typename T>
         inline _int64_t to_int64(const T v)
         {
            const typename details::number_type<T>::type num_type;
            return to_int64_impl(v, num_type);
         }

         template <typename T>
         inline bool is_nan(const T v)
         {
            const typename details::number_type<T>::type num_type;
            return is_nan_impl(v, num_type);
         }

         template <typename T>
         inline T min(const T v0, const T v1)
         {
            const typename details::number_type<T>::type num_type;
            return min_impl(v0, v1, num_type);
         }

         template <typename T>
         inline T max(const T v0, const T v1)
         {
            const typename details::number_type<T>::type num_type;
            return max_impl(v0, v1, num_type);
         }

         template <typename T>
         inline T equal(const T v0, const T v1)
         {
            const typename details::number_type<T>::type num_type;
            return equal_impl(v0, v1, num_type);
         }

         template <typename T>
         inline T nequal(const T v0, const T v1)
         {
            const typename details::number_type<T>::type num_type;
            return nequal_impl(v0, v1, num_type);
         }

         template <typename T>
         inline T modulus(const T v0, const T v1)
         {
            const typename details::number_type<T>::type num_type;
            return modulus_impl(v0, v1, num_type);
         }

         template <typename T>
         inline T pow(const T v0, const T v1)
         {
            const typename details::number_type<T>::type num_type;
            return pow_impl(v0, v1, num_type);
         }

         template <typename T>
         inline T logn(const T v0, const T v1)
         {
            const typename details::number_type<T>::type num_type;
            return logn_impl(v0, v1, num_type);
         }

         template <typename T>
         inline T root(const T v0, const T v1)
         {
            const typename details::number_type<T>::type num_type;
            return root_impl(v0, v1, num_type);
         }

         template <typename T>
         inline T roundn(const T v0, const T v1)
         {
            const typename details::number_type<T>::type num_type;
            return roundn_impl(v0, v1, num_type);
         }

         template <typename T>
         inline T hypot(const T v0, const T v1)
         {
            const typename details::number_type<T>::type num_type;
            return hypot_impl(v0, v1, num_type);
         }

         template <typename T>
         inline T atan2(const T v0, const T v1)
         {
            const typename details::number_type<T>::type num_type;
            return atan2_impl(v0, v1, num_type);
         }

         template <typename T>
         inline T shr(const T v0, const T v1)
         {
            const typename details::number_type<T>::type num_type;
            return shr_impl(v0, v1, num_type);
         }

         template <typename T>
         inline T shl(const T v0, const T v1)
         {
            const typename details::number_type<T>::type num_type;
            return shl_impl(v0, v1, num_type);
         }

         template <typename T>
         inline T and_opr(const T v0, const T v1)
         {
            const typename details::number_type<T>::type num_type;
            return and_impl(v0, v1, num_type);
         }

         template <typename T>
         inline T nand_opr(const T v0, const T v1)
         {
            const typename details::number_type<T>::type num_type;
            return nand_impl(v0, v1, num_type);
         }

         template <typename T>
         inline T or_opr(const T v0, const T v1)
         {
            const typename details::number_type<T>::type num_type;
            return or_impl(v0, v1, num_type);
         }

         template <typename T>
         inline T nor_opr(const T v0, const T v1)
         {
            const typename details::number_type<T>::type num_type;
            return nor_impl(v0, v1, num_type);
         }

         template <typename T>
         inline T xor_opr(const T v0, const T v1)
         {
            const typename details::number_type<T>::type num_type;
            return xor_impl(v0, v1, num_type);
         }

         template <typename T>
         inline T xnor_opr(const T v0, const T v1)
         {
            const typename details::number_type<T>::type num_type;
            return xnor_impl(v0, v1, num_type);
         }

         template <typename T>
         inline bool is_integer(const T v)
         {
            const typename details::number_type<T>::type num_type;
            return is_integer_impl(v, num_type);
         }

         template <typename T, unsigned int N>
         struct fast_exp
         {
            static inline T result(T v)
            {
               unsigned int k = N;
               T l = T(1);

               while (k)
               {
                  if (1 == (k % 2))
                  {
                     l *= v;
                     --k;
                  }

                  v *= v;
                  k /= 2;
               }

               return l;
            }
         };

         template <typename T> struct fast_exp<T,10> { static inline T result(const T v) { T v_5 = fast_exp<T,5>::result(v); return v_5 * v_5; } };
         template <typename T> struct fast_exp<T, 9> { static inline T result(const T v) { return fast_exp<T,8>::result(v) * v; } };
         template <typename T> struct fast_exp<T, 8> { static inline T result(const T v) { T v_4 = fast_exp<T,4>::result(v); return v_4 * v_4; } };
         template <typename T> struct fast_exp<T, 7> { static inline T result(const T v) { return fast_exp<T,6>::result(v) * v; } };
         template <typename T> struct fast_exp<T, 6> { static inline T result(const T v) { T v_3 = fast_exp<T,3>::result(v); return v_3 * v_3; } };
         template <typename T> struct fast_exp<T, 5> { static inline T result(const T v) { return fast_exp<T,4>::result(v) * v; } };
         template <typename T> struct fast_exp<T, 4> { static inline T result(const T v) { T v_2 = v * v; return v_2 * v_2; } };
         template <typename T> struct fast_exp<T, 3> { static inline T result(const T v) { return v * v * v; } };
         template <typename T> struct fast_exp<T, 2> { static inline T result(const T v) { return v * v;     } };
         template <typename T> struct fast_exp<T, 1> { static inline T result(const T v) { return v;         } };
         template <typename T> struct fast_exp<T, 0> { static inline T result(const T  ) { return T(1);      } };

         #define exprtk_define_unary_function(FunctionName)        \
         template <typename T>                                     \
         inline T FunctionName (const T v)                         \
         {                                                         \
            const typename details::number_type<T>::type num_type; \
            return  FunctionName##_impl(v,num_type);               \
         }                                                         \

         exprtk_define_unary_function(abs  )
         exprtk_define_unary_function(acos )
         exprtk_define_unary_function(acosh)
         exprtk_define_unary_function(asin )
         exprtk_define_unary_function(asinh)
         exprtk_define_unary_function(atan )
         exprtk_define_unary_function(atanh)
         exprtk_define_unary_function(ceil )
         exprtk_define_unary_function(cos  )
         exprtk_define_unary_function(cosh )
         exprtk_define_unary_function(exp  )
         exprtk_define_unary_function(expm1)
         exprtk_define_unary_function(floor)
         exprtk_define_unary_function(log  )
         exprtk_define_unary_function(log10)
         exprtk_define_unary_function(log2 )
         exprtk_define_unary_function(log1p)
         exprtk_define_unary_function(neg  )
         exprtk_define_unary_function(pos  )
         exprtk_define_unary_function(round)
         exprtk_define_unary_function(sin  )
         exprtk_define_unary_function(sinc )
         exprtk_define_unary_function(sinh )
         exprtk_define_unary_function(sqrt )
         exprtk_define_unary_function(tan  )
         exprtk_define_unary_function(tanh )
         exprtk_define_unary_function(cot  )
         exprtk_define_unary_function(sec  )
         exprtk_define_unary_function(csc  )
         exprtk_define_unary_function(r2d  )
         exprtk_define_unary_function(d2r  )
         exprtk_define_unary_function(d2g  )
         exprtk_define_unary_function(g2d  )
         exprtk_define_unary_function(notl )
         exprtk_define_unary_function(sgn  )
         exprtk_define_unary_function(erf  )
         exprtk_define_unary_function(erfc )
         exprtk_define_unary_function(ncdf )
         exprtk_define_unary_function(frac )
         exprtk_define_unary_function(trunc)
         #undef exprtk_define_unary_function
      }

      template <typename T>
      inline T compute_pow10(T d, const int exponent)
      {
         static const double fract10[] =
         {
           0.0,
           1.0E+001, 1.0E+002, 1.0E+003, 1.0E+004, 1.0E+005, 1.0E+006, 1.0E+007, 1.0E+008, 1.0E+009, 1.0E+010,
           1.0E+011, 1.0E+012, 1.0E+013, 1.0E+014, 1.0E+015, 1.0E+016, 1.0E+017, 1.0E+018, 1.0E+019, 1.0E+020,
           1.0E+021, 1.0E+022, 1.0E+023, 1.0E+024, 1.0E+025, 1.0E+026, 1.0E+027, 1.0E+028, 1.0E+029, 1.0E+030,
           1.0E+031, 1.0E+032, 1.0E+033, 1.0E+034, 1.0E+035, 1.0E+036, 1.0E+037, 1.0E+038, 1.0E+039, 1.0E+040,
           1.0E+041, 1.0E+042, 1.0E+043, 1.0E+044, 1.0E+045, 1.0E+046, 1.0E+047, 1.0E+048, 1.0E+049, 1.0E+050,
           1.0E+051, 1.0E+052, 1.0E+053, 1.0E+054, 1.0E+055, 1.0E+056, 1.0E+057, 1.0E+058, 1.0E+059, 1.0E+060,
           1.0E+061, 1.0E+062, 1.0E+063, 1.0E+064, 1.0E+065, 1.0E+066, 1.0E+067, 1.0E+068, 1.0E+069, 1.0E+070,
           1.0E+071, 1.0E+072, 1.0E+073, 1.0E+074, 1.0E+075, 1.0E+076, 1.0E+077, 1.0E+078, 1.0E+079, 1.0E+080,
           1.0E+081, 1.0E+082, 1.0E+083, 1.0E+084, 1.0E+085, 1.0E+086, 1.0E+087, 1.0E+088, 1.0E+089, 1.0E+090,
           1.0E+091, 1.0E+092, 1.0E+093, 1.0E+094, 1.0E+095, 1.0E+096, 1.0E+097, 1.0E+098, 1.0E+099, 1.0E+100,
           1.0E+101, 1.0E+102, 1.0E+103, 1.0E+104, 1.0E+105, 1.0E+106, 1.0E+107, 1.0E+108, 1.0E+109, 1.0E+110,
           1.0E+111, 1.0E+112, 1.0E+113, 1.0E+114, 1.0E+115, 1.0E+116, 1.0E+117, 1.0E+118, 1.0E+119, 1.0E+120,
           1.0E+121, 1.0E+122, 1.0E+123, 1.0E+124, 1.0E+125, 1.0E+126, 1.0E+127, 1.0E+128, 1.0E+129, 1.0E+130,
           1.0E+131, 1.0E+132, 1.0E+133, 1.0E+134, 1.0E+135, 1.0E+136, 1.0E+137, 1.0E+138, 1.0E+139, 1.0E+140,
           1.0E+141, 1.0E+142, 1.0E+143, 1.0E+144, 1.0E+145, 1.0E+146, 1.0E+147, 1.0E+148, 1.0E+149, 1.0E+150,
           1.0E+151, 1.0E+152, 1.0E+153, 1.0E+154, 1.0E+155, 1.0E+156, 1.0E+157, 1.0E+158, 1.0E+159, 1.0E+160,
           1.0E+161, 1.0E+162, 1.0E+163, 1.0E+164, 1.0E+165, 1.0E+166, 1.0E+167, 1.0E+168, 1.0E+169, 1.0E+170,
           1.0E+171, 1.0E+172, 1.0E+173, 1.0E+174, 1.0E+175, 1.0E+176, 1.0E+177, 1.0E+178, 1.0E+179, 1.0E+180,
           1.0E+181, 1.0E+182, 1.0E+183, 1.0E+184, 1.0E+185, 1.0E+186, 1.0E+187, 1.0E+188, 1.0E+189, 1.0E+190,
           1.0E+191, 1.0E+192, 1.0E+193, 1.0E+194, 1.0E+195, 1.0E+196, 1.0E+197, 1.0E+198, 1.0E+199, 1.0E+200,
           1.0E+201, 1.0E+202, 1.0E+203, 1.0E+204, 1.0E+205, 1.0E+206, 1.0E+207, 1.0E+208, 1.0E+209, 1.0E+210,
           1.0E+211, 1.0E+212, 1.0E+213, 1.0E+214, 1.0E+215, 1.0E+216, 1.0E+217, 1.0E+218, 1.0E+219, 1.0E+220,
           1.0E+221, 1.0E+222, 1.0E+223, 1.0E+224, 1.0E+225, 1.0E+226, 1.0E+227, 1.0E+228, 1.0E+229, 1.0E+230,
           1.0E+231, 1.0E+232, 1.0E+233, 1.0E+234, 1.0E+235, 1.0E+236, 1.0E+237, 1.0E+238, 1.0E+239, 1.0E+240,
           1.0E+241, 1.0E+242, 1.0E+243, 1.0E+244, 1.0E+245, 1.0E+246, 1.0E+247, 1.0E+248, 1.0E+249, 1.0E+250,
           1.0E+251, 1.0E+252, 1.0E+253, 1.0E+254, 1.0E+255, 1.0E+256, 1.0E+257, 1.0E+258, 1.0E+259, 1.0E+260,
           1.0E+261, 1.0E+262, 1.0E+263, 1.0E+264, 1.0E+265, 1.0E+266, 1.0E+267, 1.0E+268, 1.0E+269, 1.0E+270,
           1.0E+271, 1.0E+272, 1.0E+273, 1.0E+274, 1.0E+275, 1.0E+276, 1.0E+277, 1.0E+278, 1.0E+279, 1.0E+280,
           1.0E+281, 1.0E+282, 1.0E+283, 1.0E+284, 1.0E+285, 1.0E+286, 1.0E+287, 1.0E+288, 1.0E+289, 1.0E+290,
           1.0E+291, 1.0E+292, 1.0E+293, 1.0E+294, 1.0E+295, 1.0E+296, 1.0E+297, 1.0E+298, 1.0E+299, 1.0E+300,
           1.0E+301, 1.0E+302, 1.0E+303, 1.0E+304, 1.0E+305, 1.0E+306, 1.0E+307, 1.0E+308
         };

         static const int fract10_size = static_cast<int>(sizeof(fract10) / sizeof(double));

         const int e = std::abs(exponent);

         if (exponent >= std::numeric_limits<T>::min_exponent10)
         {
            if (e < fract10_size)
            {
               if (exponent > 0)
                  return T(d * fract10[e]);
               else
                  return T(d / fract10[e]);
            }
            else
               return T(d * std::pow(10.0, 10.0 * exponent));
         }
         else
         {
                     d /= T(fract10[           -std::numeric_limits<T>::min_exponent10]);
            return T(d /    fract10[-exponent + std::numeric_limits<T>::min_exponent10]);
         }
      }

      template <typename Iterator, typename T>
      inline bool string_to_type_converter_impl_ref(Iterator& itr, const Iterator end, T& result)
      {
         if (itr == end)
            return false;

         const bool negative = ('-' == (*itr));

         if (negative || ('+' == (*itr)))
         {
            if (end == ++itr)
               return false;
         }

         static const uchar_t zero = static_cast<uchar_t>('0');

         while ((end != itr) && (zero == (*itr))) ++itr;

         bool return_result = true;
         unsigned int digit = 0;
         const std::size_t length = static_cast<std::size_t>(std::distance(itr,end));

         if (length <= 4)
         {
            exprtk_disable_fallthrough_begin
            switch (length)
            {
               #ifdef exprtk_use_lut

               #define exprtk_process_digit                          \
               if ((digit = details::digit_table[(int)*itr++]) < 10) \
                  result = result * 10 + (digit);                    \
               else                                                  \
               {                                                     \
                  return_result = false;                             \
                  break;                                             \
               }                                                     \

               #else

               #define exprtk_process_digit         \
               if ((digit = (*itr++ - zero)) < 10)  \
                  result = result * T(10) + digit;  \
               else                                 \
               {                                    \
                  return_result = false;            \
                  break;                            \
               }                                    \

               #endif

               case 4 : exprtk_process_digit
               case 3 : exprtk_process_digit
               case 2 : exprtk_process_digit
               case 1 : if ((digit = (*itr - zero))>= 10)
                        {
                           digit = 0;
                           return_result = false;
                        }

               #undef exprtk_process_digit
            }
            exprtk_disable_fallthrough_end
         }
         else
            return_result = false;

         if (length && return_result)
         {
            result = result * 10 + static_cast<T>(digit);
            ++itr;
         }

         result = negative ? -result : result;
         return return_result;
      }

      template <typename Iterator, typename T>
      static inline bool parse_nan(Iterator& itr, const Iterator end, T& t)
      {
         typedef typename std::iterator_traits<Iterator>::value_type type;

         static const std::size_t nan_length = 3;

         if (std::distance(itr,end) != static_cast<int>(nan_length))
            return false;

         if (static_cast<type>('n') == (*itr))
         {
            if (
                 (static_cast<type>('a') != *(itr + 1)) ||
                 (static_cast<type>('n') != *(itr + 2))
               )
            {
               return false;
            }
         }
         else if (
                   (static_cast<type>('A') != *(itr + 1)) ||
                   (static_cast<type>('N') != *(itr + 2))
                 )
         {
            return false;
         }

         t = std::numeric_limits<T>::quiet_NaN();

         return true;
      }

      template <typename Iterator, typename T>
      static inline bool parse_inf(Iterator& itr, const Iterator end, T& t, const bool negative)
      {
         static const char_t inf_uc[] = "INFINITY";
         static const char_t inf_lc[] = "infinity";
         static const std::size_t inf_length = 8;

         const std::size_t length = static_cast<std::size_t>(std::distance(itr,end));

         if ((3 != length) && (inf_length != length))
            return false;

         char_cptr inf_itr = ('i' == (*itr)) ? inf_lc : inf_uc;

         while (end != itr)
         {
            if (*inf_itr == static_cast<char_t>(*itr))
            {
               ++itr;
               ++inf_itr;
               continue;
            }
            else
               return false;
         }

         if (negative)
            t = -std::numeric_limits<T>::infinity();
         else
            t =  std::numeric_limits<T>::infinity();

         return true;
      }

      template <typename T>
      inline bool valid_exponent(const int exponent, numeric::details::real_type_tag)
      {
         using namespace details::numeric;
         return (numeric_info<T>::min_exp <= exponent) && (exponent <= numeric_info<T>::max_exp);
      }

      template <typename Iterator, typename T>
      inline bool string_to_real(Iterator& itr_external, const Iterator end, T& t, numeric::details::real_type_tag)
      {
         if (end == itr_external) return false;

         Iterator itr = itr_external;

         T d = T(0);

         const bool negative = ('-' == (*itr));

         if (negative || '+' == (*itr))
         {
            if (end == ++itr)
               return false;
         }

         bool instate = false;

         static const char_t zero = static_cast<uchar_t>('0');

         #define parse_digit_1(d)          \
         if ((digit = (*itr - zero)) < 10) \
            { d = d * T(10) + digit; }     \
         else                              \
            { break; }                     \
         if (end == ++itr) break;          \

         #define parse_digit_2(d)          \
         if ((digit = (*itr - zero)) < 10) \
            { d = d * T(10) + digit; }     \
         else                              \
            { break; }                     \
            ++itr;                         \

         if ('.' != (*itr))
         {
            const Iterator curr = itr;

            while ((end != itr) && (zero == (*itr))) ++itr;

            while (end != itr)
            {
               unsigned int digit;
               parse_digit_1(d)
               parse_digit_1(d)
               parse_digit_2(d)
            }

            if (curr != itr) instate = true;
         }

         int exponent = 0;

         if (end != itr)
         {
            if ('.' == (*itr))
            {
               const Iterator curr = ++itr;
               T tmp_d = T(0);

               while (end != itr)
               {
                  unsigned int digit;
                  parse_digit_1(tmp_d)
                  parse_digit_1(tmp_d)
                  parse_digit_2(tmp_d)
               }

               if (curr != itr)
               {
                  instate = true;

                  const int frac_exponent = static_cast<int>(-std::distance(curr, itr));

                  if (!valid_exponent<T>(frac_exponent, numeric::details::real_type_tag()))
                     return false;

                  d += compute_pow10(tmp_d, frac_exponent);
               }

               #undef parse_digit_1
               #undef parse_digit_2
            }

            if (end != itr)
            {
               typename std::iterator_traits<Iterator>::value_type c = (*itr);

               if (('e' == c) || ('E' == c))
               {
                  int exp = 0;

                  if (!details::string_to_type_converter_impl_ref(++itr, end, exp))
                  {
                     if (end == itr)
                        return false;
                     else
                        c = (*itr);
                  }

                  exponent += exp;
               }

               if (end != itr)
               {
                  if (('f' == c) || ('F' == c) || ('l' == c) || ('L' == c))
                     ++itr;
                  else if ('#' == c)
                  {
                     if (end == ++itr)
                        return false;
                     else if (('I' <= (*itr)) && ((*itr) <= 'n'))
                     {
                        if (('i' == (*itr)) || ('I' == (*itr)))
                        {
                           return parse_inf(itr, end, t, negative);
                        }
                        else if (('n' == (*itr)) || ('N' == (*itr)))
                        {
                           return parse_nan(itr, end, t);
                        }
                        else
                           return false;
                     }
                     else
                        return false;
                  }
                  else if (('I' <= (*itr)) && ((*itr) <= 'n'))
                  {
                     if (('i' == (*itr)) || ('I' == (*itr)))
                     {
                        return parse_inf(itr, end, t, negative);
                     }
                     else if (('n' == (*itr)) || ('N' == (*itr)))
                     {
                        return parse_nan(itr, end, t);
                     }
                     else
                        return false;
                  }
                  else
                     return false;
               }
            }
         }

         if ((end != itr) || (!instate))
            return false;
         else if (!valid_exponent<T>(exponent, numeric::details::real_type_tag()))
            return false;
         else if (exponent)
            d = compute_pow10(d,exponent);

         t = static_cast<T>((negative) ? -d : d);
         return true;
      }

      template <typename T>
      inline bool string_to_real(const std::string& s, T& t)
      {
         const typename numeric::details::number_type<T>::type num_type;

         char_cptr begin = s.data();
         char_cptr end   = s.data() + s.size();

         return string_to_real(begin, end, t, num_type);
      }

      template <typename T>
      struct functor_t
      {
         /*
            Note: The following definitions for Type, may require tweaking
                  based on the compiler and target architecture. The benchmark
                  should provide enough information to make the right choice.
         */
         //typedef T Type;
         //typedef const T Type;
         typedef const T& Type;
         typedef       T& RefType;
         typedef T (*qfunc_t)(Type t0, Type t1, Type t2, Type t3);
         typedef T (*tfunc_t)(Type t0, Type t1, Type t2);
         typedef T (*bfunc_t)(Type t0, Type t1);
         typedef T (*ufunc_t)(Type t0);
      };

   } // namespace details

   struct loop_runtime_check
   {
      enum loop_types
      {
         e_invalid           = 0,
         e_for_loop          = 1,
         e_while_loop        = 2,
         e_repeat_until_loop = 4,
         e_all_loops         = 7
      };

      enum violation_type
      {
          e_unknown         = 0,
          e_iteration_count = 1,
          e_timeout         = 2
      };

      loop_types loop_set;

      loop_runtime_check()
      : loop_set(e_invalid)
      , max_loop_iterations(0)
      {}

      details::_uint64_t max_loop_iterations;

      struct violation_context
      {
         loop_types loop;
         violation_type violation;
         details::_uint64_t iteration_count;
      };

      virtual void handle_runtime_violation(const violation_context&)
      {
         throw std::runtime_error("ExprTk Loop run-time violation.");
      }

      virtual ~loop_runtime_check() {}
   };

   typedef loop_runtime_check* loop_runtime_check_ptr;

   namespace lexer
   {
      struct token
      {
         enum token_type
         {
            e_none        =   0, e_error       =   1, e_err_symbol  =   2,
            e_err_number  =   3, e_err_string  =   4, e_err_sfunc   =   5,
            e_eof         =   6, e_number      =   7, e_symbol      =   8,
            e_string      =   9, e_assign      =  10, e_addass      =  11,
            e_subass      =  12, e_mulass      =  13, e_divass      =  14,
            e_modass      =  15, e_shr         =  16, e_shl         =  17,
            e_lte         =  18, e_ne          =  19, e_gte         =  20,
            e_swap        =  21, e_lt          = '<', e_gt          = '>',
            e_eq          = '=', e_rbracket    = ')', e_lbracket    = '(',
            e_rsqrbracket = ']', e_lsqrbracket = '[', e_rcrlbracket = '}',
            e_lcrlbracket = '{', e_comma       = ',', e_add         = '+',
            e_sub         = '-', e_div         = '/', e_mul         = '*',
            e_mod         = '%', e_pow         = '^', e_colon       = ':',
            e_ternary     = '?'
         };

         token()
         : type(e_none)
         , value("")
         , position(std::numeric_limits<std::size_t>::max())
         {}

         void clear()
         {
            type     = e_none;
            value    = "";
            position = std::numeric_limits<std::size_t>::max();
         }

         template <typename Iterator>
         inline token& set_operator(const token_type tt,
                                    const Iterator begin, const Iterator end,
                                    const Iterator base_begin = Iterator(0))
         {
            type = tt;
            value.assign(begin,end);
            if (base_begin)
               position = static_cast<std::size_t>(std::distance(base_begin,begin));
            return (*this);
         }

         template <typename Iterator>
         inline token& set_symbol(const Iterator begin, const Iterator end, const Iterator base_begin = Iterator(0))
         {
            type = e_symbol;
            value.assign(begin,end);
            if (base_begin)
               position = static_cast<std::size_t>(std::distance(base_begin,begin));
            return (*this);
         }

         template <typename Iterator>
         inline token& set_numeric(const Iterator begin, const Iterator end, const Iterator base_begin = Iterator(0))
         {
            type = e_number;
            value.assign(begin,end);
            if (base_begin)
               position = static_cast<std::size_t>(std::distance(base_begin,begin));
            return (*this);
         }

         template <typename Iterator>
         inline token& set_string(const Iterator begin, const Iterator end, const Iterator base_begin = Iterator(0))
         {
            type = e_string;
            value.assign(begin,end);
            if (base_begin)
               position = static_cast<std::size_t>(std::distance(base_begin,begin));
            return (*this);
         }

         inline token& set_string(const std::string& s, const std::size_t p)
         {
            type     = e_string;
            value    = s;
            position = p;
            return (*this);
         }

         template <typename Iterator>
         inline token& set_error(const token_type et,
                                 const Iterator begin, const Iterator end,
                                 const Iterator base_begin = Iterator(0))
         {
            if (
                 (e_error      == et) ||
                 (e_err_symbol == et) ||
                 (e_err_number == et) ||
                 (e_err_string == et) ||
                 (e_err_sfunc  == et)
               )
            {
               type = et;
            }
            else
               type = e_error;

            value.assign(begin,end);

            if (base_begin)
               position = static_cast<std::size_t>(std::distance(base_begin,begin));

            return (*this);
         }

         static inline std::string to_str(token_type t)
         {
            switch (t)
            {
               case e_none        : return "NONE";
               case e_error       : return "ERROR";
               case e_err_symbol  : return "ERROR_SYMBOL";
               case e_err_number  : return "ERROR_NUMBER";
               case e_err_string  : return "ERROR_STRING";
               case e_eof         : return "EOF";
               case e_number      : return "NUMBER";
               case e_symbol      : return "SYMBOL";
               case e_string      : return "STRING";
               case e_assign      : return ":=";
               case e_addass      : return "+=";
               case e_subass      : return "-=";
               case e_mulass      : return "*=";
               case e_divass      : return "/=";
               case e_modass      : return "%=";
               case e_shr         : return ">>";
               case e_shl         : return "<<";
               case e_lte         : return "<=";
               case e_ne          : return "!=";
               case e_gte         : return ">=";
               case e_lt          : return "<";
               case e_gt          : return ">";
               case e_eq          : return "=";
               case e_rbracket    : return ")";
               case e_lbracket    : return "(";
               case e_rsqrbracket : return "]";
               case e_lsqrbracket : return "[";
               case e_rcrlbracket : return "}";
               case e_lcrlbracket : return "{";
               case e_comma       : return ",";
               case e_add         : return "+";
               case e_sub         : return "-";
               case e_div         : return "/";
               case e_mul         : return "*";
               case e_mod         : return "%";
               case e_pow         : return "^";
               case e_colon       : return ":";
               case e_ternary     : return "?";
               case e_swap        : return "<=>";
               default            : return "UNKNOWN";
            }
         }

         inline bool is_error() const
         {
            return (
                     (e_error      == type) ||
                     (e_err_symbol == type) ||
                     (e_err_number == type) ||
                     (e_err_string == type) ||
                     (e_err_sfunc  == type)
                   );
         }

         token_type type;
         std::string value;
         std::size_t position;
      };

      class generator
      {
      public:

         typedef token token_t;
         typedef std::vector<token_t> token_list_t;
         typedef token_list_t::iterator token_list_itr_t;
         typedef details::char_t char_t;

         generator()
         : base_itr_(0)
         , s_itr_   (0)
         , s_end_   (0)
         {
            clear();
         }

         inline void clear()
         {
            base_itr_ = 0;
            s_itr_    = 0;
            s_end_    = 0;
            token_list_.clear();
            token_itr_ = token_list_.end();
            store_token_itr_ = token_list_.end();
         }

         inline bool process(const std::string& str)
         {
            base_itr_ = str.data();
            s_itr_    = str.data();
            s_end_    = str.data() + str.size();

            eof_token_.set_operator(token_t::e_eof,s_end_,s_end_,base_itr_);
            token_list_.clear();

            while (!is_end(s_itr_))
            {
               scan_token();

               if (!token_list_.empty() && token_list_.back().is_error())
                  return false;
            }

            return true;
         }

         inline bool empty() const
         {
            return token_list_.empty();
         }

         inline std::size_t size() const
         {
            return token_list_.size();
         }

         inline void begin()
         {
            token_itr_ = token_list_.begin();
            store_token_itr_ = token_list_.begin();
         }

         inline void store()
         {
            store_token_itr_ = token_itr_;
         }

         inline void restore()
         {
            token_itr_ = store_token_itr_;
         }

         inline token_t& next_token()
         {
            if (token_list_.end() != token_itr_)
            {
               return *token_itr_++;
            }
            else
               return eof_token_;
         }

         inline token_t& peek_next_token()
         {
            if (token_list_.end() != token_itr_)
            {
               return *token_itr_;
            }
            else
               return eof_token_;
         }

         inline token_t& operator[](const std::size_t& index)
         {
            if (index < token_list_.size())
               return token_list_[index];
            else
               return eof_token_;
         }

         inline token_t operator[](const std::size_t& index) const
         {
            if (index < token_list_.size())
               return token_list_[index];
            else
               return eof_token_;
         }

         inline bool finished() const
         {
            return (token_list_.end() == token_itr_);
         }

         inline void insert_front(token_t::token_type tk_type)
         {
            if (
                 !token_list_.empty() &&
                 (token_list_.end() != token_itr_)
               )
            {
               token_t t = *token_itr_;

               t.type     = tk_type;
               token_itr_ = token_list_.insert(token_itr_,t);
            }
         }

         inline std::string substr(const std::size_t& begin, const std::size_t& end) const
         {
            const details::char_cptr begin_itr = ((base_itr_ + begin) < s_end_) ? (base_itr_ + begin) : s_end_;
            const details::char_cptr end_itr   = ((base_itr_ + end  ) < s_end_) ? (base_itr_ + end  ) : s_end_;

            return std::string(begin_itr,end_itr);
         }

         inline std::string remaining() const
         {
            if (finished())
               return "";
            else if (token_list_.begin() != token_itr_)
               return std::string(base_itr_ + (token_itr_ - 1)->position, s_end_);
            else
               return std::string(base_itr_ + token_itr_->position, s_end_);
         }

      private:

         inline bool is_end(details::char_cptr itr) const
         {
            return (s_end_ == itr);
         }

         inline bool is_comment_start(details::char_cptr) const
         {
            return false;
         }

         inline void skip_whitespace()
         {
            while (!is_end(s_itr_) && details::is_whitespace(*s_itr_))
            {
               ++s_itr_;
            }
         }

         inline void skip_comments()
         {
         }

         inline void scan_token()
         {
            if (details::is_whitespace(*s_itr_))
            {
               skip_whitespace();
               return;
            }
            else if (is_comment_start(s_itr_))
            {
               skip_comments();
               return;
            }
            else if (details::is_operator_char(*s_itr_))
            {
               scan_operator();
               return;
            }
            else if (details::is_letter(*s_itr_))
            {
               scan_symbol();
               return;
            }
            else if (details::is_digit((*s_itr_)) || ('.' == (*s_itr_)))
            {
               scan_number();
               return;
            }
            else if ('$' == (*s_itr_))
            {
               scan_special_function();
               return;
            }
            else if ('~' == (*s_itr_))
            {
               token_t t;
               t.set_symbol(s_itr_, s_itr_ + 1, base_itr_);
               token_list_.push_back(t);
               ++s_itr_;
               return;
            }
            else
            {
               token_t t;
               t.set_error(token::e_error, s_itr_, s_itr_ + 2, base_itr_);
               token_list_.push_back(t);
               ++s_itr_;
            }
         }

         inline void scan_operator()
         {
            token_t t;

            const char_t c0 = s_itr_[0];

            if (!is_end(s_itr_ + 1))
            {
               const char_t c1 = s_itr_[1];

               if (!is_end(s_itr_ + 2))
               {
                  const char_t c2 = s_itr_[2];

                  if ((c0 == '<') && (c1 == '=') && (c2 == '>'))
                  {
                     t.set_operator(token_t::e_swap, s_itr_, s_itr_ + 3, base_itr_);
                     token_list_.push_back(t);
                     s_itr_ += 3;
                     return;
                  }
               }

               token_t::token_type ttype = token_t::e_none;

               if      ((c0 == '<') && (c1 == '=')) ttype = token_t::e_lte;
               else if ((c0 == '>') && (c1 == '=')) ttype = token_t::e_gte;
               else if ((c0 == '<') && (c1 == '>')) ttype = token_t::e_ne;
               else if ((c0 == '!') && (c1 == '=')) ttype = token_t::e_ne;
               else if ((c0 == '=') && (c1 == '=')) ttype = token_t::e_eq;
               else if ((c0 == ':') && (c1 == '=')) ttype = token_t::e_assign;
               else if ((c0 == '<') && (c1 == '<')) ttype = token_t::e_shl;
               else if ((c0 == '>') && (c1 == '>')) ttype = token_t::e_shr;
               else if ((c0 == '+') && (c1 == '=')) ttype = token_t::e_addass;
               else if ((c0 == '-') && (c1 == '=')) ttype = token_t::e_subass;
               else if ((c0 == '*') && (c1 == '=')) ttype = token_t::e_mulass;
               else if ((c0 == '/') && (c1 == '=')) ttype = token_t::e_divass;
               else if ((c0 == '%') && (c1 == '=')) ttype = token_t::e_modass;

               if (token_t::e_none != ttype)
               {
                  t.set_operator(ttype, s_itr_, s_itr_ + 2, base_itr_);
                  token_list_.push_back(t);
                  s_itr_ += 2;
                  return;
               }
            }

            if ('<' == c0)
               t.set_operator(token_t::e_lt , s_itr_, s_itr_ + 1, base_itr_);
            else if ('>' == c0)
               t.set_operator(token_t::e_gt , s_itr_, s_itr_ + 1, base_itr_);
            else if (';' == c0)
               t.set_operator(token_t::e_eof, s_itr_, s_itr_ + 1, base_itr_);
            else if ('&' == c0)
               t.set_symbol(s_itr_, s_itr_ + 1, base_itr_);
            else if ('|' == c0)
               t.set_symbol(s_itr_, s_itr_ + 1, base_itr_);
            else
               t.set_operator(token_t::token_type(c0), s_itr_, s_itr_ + 1, base_itr_);

            token_list_.push_back(t);
            ++s_itr_;
         }

         inline void scan_symbol()
         {
            details::char_cptr initial_itr = s_itr_;

            while (!is_end(s_itr_))
            {
               if (!details::is_letter_or_digit(*s_itr_) && ('_' != (*s_itr_)))
               {
                  if ('.' != (*s_itr_))
                     break;
                  /*
                     Permit symbols that contain a 'dot'
                     Allowed   : abc.xyz, a123.xyz, abc.123, abc_.xyz a123_.xyz abc._123
                     Disallowed: .abc, abc.<white-space>, abc.<eof>, abc.<operator +,-,*,/...>
                  */
                  if (
                       (s_itr_ != initial_itr)                     &&
                       !is_end(s_itr_ + 1)                         &&
                       !details::is_letter_or_digit(*(s_itr_ + 1)) &&
                       ('_' != (*(s_itr_ + 1)))
                     )
                     break;
               }

               ++s_itr_;
            }

            token_t t;
            t.set_symbol(initial_itr,s_itr_,base_itr_);
            token_list_.push_back(t);
         }

         inline void scan_number()
         {
            /*
               Attempt to match a valid numeric value in one of the following formats:
               (01) 123456
               (02) 123456.
               (03) 123.456
               (04) 123.456e3
               (05) 123.456E3
               (06) 123.456e+3
               (07) 123.456E+3
               (08) 123.456e-3
               (09) 123.456E-3
               (00) .1234
               (11) .1234e3
               (12) .1234E+3
               (13) .1234e+3
               (14) .1234E-3
               (15) .1234e-3
            */

            details::char_cptr initial_itr = s_itr_;
            bool dot_found                 = false;
            bool e_found                   = false;
            bool post_e_sign_found         = false;
            bool post_e_digit_found        = false;
            token_t t;

            while (!is_end(s_itr_))
            {
               if ('.' == (*s_itr_))
               {
                  if (dot_found)
                  {
                     t.set_error(token::e_err_number, initial_itr, s_itr_, base_itr_);
                     token_list_.push_back(t);

                     return;
                  }

                  dot_found = true;
                  ++s_itr_;

                  continue;
               }
               else if ('e' == std::tolower(*s_itr_))
               {
                  const char_t& c = *(s_itr_ + 1);

                  if (is_end(s_itr_ + 1))
                  {
                     t.set_error(token::e_err_number, initial_itr, s_itr_, base_itr_);
                     token_list_.push_back(t);

                     return;
                  }
                  else if (
                            ('+' != c) &&
                            ('-' != c) &&
                            !details::is_digit(c)
                          )
                  {
                     t.set_error(token::e_err_number, initial_itr, s_itr_, base_itr_);
                     token_list_.push_back(t);

                     return;
                  }

                  e_found = true;
                  ++s_itr_;

                  continue;
               }
               else if (e_found && details::is_sign(*s_itr_) && !post_e_digit_found)
               {
                  if (post_e_sign_found)
                  {
                     t.set_error(token::e_err_number, initial_itr, s_itr_, base_itr_);
                     token_list_.push_back(t);

                     return;
                  }

                  post_e_sign_found = true;
                  ++s_itr_;

                  continue;
               }
               else if (e_found && details::is_digit(*s_itr_))
               {
                  post_e_digit_found = true;
                  ++s_itr_;

                  continue;
               }
               else if (('.' != (*s_itr_)) && !details::is_digit(*s_itr_))
                  break;
               else
                  ++s_itr_;
            }

            t.set_numeric(initial_itr, s_itr_, base_itr_);
            token_list_.push_back(t);

            return;
         }

         inline void scan_special_function()
         {
            details::char_cptr initial_itr = s_itr_;
            token_t t;

            // $fdd(x,x,x) = at least 11 chars
            if (std::distance(s_itr_,s_end_) < 11)
            {
               t.set_error(
                  token::e_err_sfunc,
                  initial_itr, std::min(initial_itr + 11, s_end_),
                  base_itr_);
               token_list_.push_back(t);

               return;
            }

            if (
                 !(('$' == *s_itr_)                       &&
                   (details::imatch  ('f',*(s_itr_ + 1))) &&
                   (details::is_digit(*(s_itr_ + 2)))     &&
                   (details::is_digit(*(s_itr_ + 3))))
               )
            {
               t.set_error(
                  token::e_err_sfunc,
                  initial_itr, std::min(initial_itr + 4, s_end_),
                  base_itr_);
               token_list_.push_back(t);

               return;
            }

            s_itr_ += 4; // $fdd = 4chars

            t.set_symbol(initial_itr, s_itr_, base_itr_);
            token_list_.push_back(t);

            return;
         }

      private:

         token_list_t       token_list_;
         token_list_itr_t   token_itr_;
         token_list_itr_t   store_token_itr_;
         token_t            eof_token_;
         details::char_cptr base_itr_;
         details::char_cptr s_itr_;
         details::char_cptr s_end_;

         friend class token_scanner;
         friend class token_modifier;
         friend class token_inserter;
         friend class token_joiner;
      }; // class generator

      class helper_interface
      {
      public:

         virtual void init()                     {              }
         virtual void reset()                    {              }
         virtual bool result()                   { return true; }
         virtual std::size_t process(generator&) { return 0;    }
         virtual ~helper_interface()             {              }
      };

      class token_scanner : public helper_interface
      {
      public:

         virtual ~token_scanner() {}

         explicit token_scanner(const std::size_t& stride)
         : stride_(stride)
         {
            if (stride > 4)
            {
               throw std::invalid_argument("token_scanner() - Invalid stride value");
            }
         }

         inline std::size_t process(generator& g) exprtk_override
         {
            if (g.token_list_.size() >= stride_)
            {
               for (std::size_t i = 0; i < (g.token_list_.size() - stride_ + 1); ++i)
               {
                  token t;

                  switch (stride_)
                  {
                     case 1 :
                              {
                                 const token& t0 = g.token_list_[i];

                                 if (!operator()(t0))
                                 {
                                    return i;
                                 }
                              }
                              break;

                     case 2 :
                              {
                                 const token& t0 = g.token_list_[i    ];
                                 const token& t1 = g.token_list_[i + 1];

                                 if (!operator()(t0, t1))
                                 {
                                    return i;
                                 }
                              }
                              break;

                     case 3 :
                              {
                                 const token& t0 = g.token_list_[i    ];
                                 const token& t1 = g.token_list_[i + 1];
                                 const token& t2 = g.token_list_[i + 2];

                                 if (!operator()(t0, t1, t2))
                                 {
                                    return i;
                                 }
                              }
                              break;

                     case 4 :
                              {
                                 const token& t0 = g.token_list_[i    ];
                                 const token& t1 = g.token_list_[i + 1];
                                 const token& t2 = g.token_list_[i + 2];
                                 const token& t3 = g.token_list_[i + 3];

                                 if (!operator()(t0, t1, t2, t3))
                                 {
                                    return i;
                                 }
                              }
                              break;
                  }
               }
            }

            return (g.token_list_.size() - stride_ + 1);
         }

         virtual bool operator() (const token&)
         {
            return false;
         }

         virtual bool operator() (const token&, const token&)
         {
            return false;
         }

         virtual bool operator() (const token&, const token&, const token&)
         {
            return false;
         }

         virtual bool operator() (const token&, const token&, const token&, const token&)
         {
            return false;
         }

      private:

         const std::size_t stride_;
      }; // class token_scanner

      class token_modifier : public helper_interface
      {
      public:

         inline std::size_t process(generator& g) exprtk_override
         {
            std::size_t changes = 0;

            for (std::size_t i = 0; i < g.token_list_.size(); ++i)
            {
               if (modify(g.token_list_[i])) changes++;
            }

            return changes;
         }

         virtual bool modify(token& t) = 0;
      };

      class token_inserter : public helper_interface
      {
      public:

         explicit token_inserter(const std::size_t& stride)
         : stride_(stride)
         {
            if (stride > 5)
            {
               throw std::invalid_argument("token_inserter() - Invalid stride value");
            }
         }

         inline std::size_t process(generator& g) exprtk_override
         {
            if (g.token_list_.empty())
               return 0;
            else if (g.token_list_.size() < stride_)
               return 0;

            std::size_t changes = 0;

            typedef std::pair<std::size_t, token> insert_t;
            std::vector<insert_t> insert_list;
            insert_list.reserve(10000);

            for (std::size_t i = 0; i < (g.token_list_.size() - stride_ + 1); ++i)
            {
               int insert_index = -1;
               token t;

               switch (stride_)
               {
                  case 1 : insert_index = insert(g.token_list_[i],t);
                           break;

                  case 2 : insert_index = insert(g.token_list_[i], g.token_list_[i + 1], t);
                           break;

                  case 3 : insert_index = insert(g.token_list_[i], g.token_list_[i + 1], g.token_list_[i + 2], t);
                           break;

                  case 4 : insert_index = insert(g.token_list_[i], g.token_list_[i + 1], g.token_list_[i + 2], g.token_list_[i + 3], t);
                           break;

                  case 5 : insert_index = insert(g.token_list_[i], g.token_list_[i + 1], g.token_list_[i + 2], g.token_list_[i + 3], g.token_list_[i + 4], t);
                           break;
               }

               if ((insert_index >= 0) && (insert_index <= (static_cast<int>(stride_) + 1)))
               {
                  insert_list.push_back(insert_t(i, t));
                  changes++;
               }
            }

            if (!insert_list.empty())
            {
               generator::token_list_t token_list;

               std::size_t insert_index = 0;

               for (std::size_t i = 0; i < g.token_list_.size(); ++i)
               {
                  token_list.push_back(g.token_list_[i]);

                  if (
                       (insert_index < insert_list.size()) &&
                       (insert_list[insert_index].first == i)
                     )
                  {
                     token_list.push_back(insert_list[insert_index].second);
                     insert_index++;
                  }
               }

               std::swap(g.token_list_,token_list);
            }

            return changes;
         }

         #define token_inserter_empty_body \
         {                                 \
            return -1;                     \
         }                                 \

         inline virtual int insert(const token&, token&)
         token_inserter_empty_body

         inline virtual int insert(const token&, const token&, token&)
         token_inserter_empty_body

         inline virtual int insert(const token&, const token&, const token&, token&)
         token_inserter_empty_body

         inline virtual int insert(const token&, const token&, const token&, const token&, token&)
         token_inserter_empty_body

         inline virtual int insert(const token&, const token&, const token&, const token&, const token&, token&)
         token_inserter_empty_body

         #undef token_inserter_empty_body

      private:

         const std::size_t stride_;
      };

      class token_joiner : public helper_interface
      {
      public:

         explicit token_joiner(const std::size_t& stride)
         : stride_(stride)
         {}

         inline std::size_t process(generator& g) exprtk_override
         {
            if (g.token_list_.empty())
               return 0;

            switch (stride_)
            {
               case 2  : return process_stride_2(g);
               case 3  : return process_stride_3(g);
               default : return 0;
            }
         }

         virtual bool join(const token&, const token&, token&)               { return false; }
         virtual bool join(const token&, const token&, const token&, token&) { return false; }

      private:

         inline std::size_t process_stride_2(generator& g)
         {
            if (g.token_list_.size() < 2)
               return 0;

            std::size_t changes = 0;

            generator::token_list_t token_list;
            token_list.reserve(10000);

            for (int i = 0;  i < static_cast<int>(g.token_list_.size() - 1); ++i)
            {
               token t;

               for ( ; ; )
               {
                  if (!join(g[i], g[i + 1], t))
                  {
                     token_list.push_back(g[i]);
                     break;
                  }

                  token_list.push_back(t);

                  ++changes;

                  i+=2;

                  if (static_cast<std::size_t>(i) >= (g.token_list_.size() - 1))
                     break;
               }
            }

            token_list.push_back(g.token_list_.back());

            assert(token_list.size() <= g.token_list_.size());

            std::swap(token_list, g.token_list_);

            return changes;
         }

         inline std::size_t process_stride_3(generator& g)
         {
            if (g.token_list_.size() < 3)
               return 0;

            std::size_t changes = 0;

            generator::token_list_t token_list;
            token_list.reserve(10000);

            for (int i = 0;  i < static_cast<int>(g.token_list_.size() - 2); ++i)
            {
               token t;

               for ( ; ; )
               {
                  if (!join(g[i], g[i + 1], g[i + 2], t))
                  {
                     token_list.push_back(g[i]);
                     break;
                  }

                  token_list.push_back(t);

                  ++changes;

                  i+=3;

                  if (static_cast<std::size_t>(i) >= (g.token_list_.size() - 2))
                     break;
               }
            }

            token_list.push_back(*(g.token_list_.begin() + g.token_list_.size() - 2));
            token_list.push_back(*(g.token_list_.begin() + g.token_list_.size() - 1));

            assert(token_list.size() <= g.token_list_.size());

            std::swap(token_list, g.token_list_);

            return changes;
         }

         const std::size_t stride_;
      };

      namespace helper
      {

         inline void dump(const lexer::generator& generator)
         {
            for (std::size_t i = 0; i < generator.size(); ++i)
            {
               const lexer::token& t = generator[i];
               printf("Token[%02d] @ %03d  %6s  -->  '%s'\n",
                      static_cast<int>(i),
                      static_cast<int>(t.position),
                      t.to_str(t.type).c_str(),
                      t.value.c_str());
            }
         }

         class commutative_inserter : public lexer::token_inserter
         {
         public:

            using lexer::token_inserter::insert;

            commutative_inserter()
            : lexer::token_inserter(2)
            {}

            inline void ignore_symbol(const std::string& symbol)
            {
               ignore_set_.insert(symbol);
            }

            inline int insert(const lexer::token& t0, const lexer::token& t1, lexer::token& new_token)
            {
               bool match         = false;
               new_token.type     = lexer::token::e_mul;
               new_token.value    = "*";
               new_token.position = t1.position;

               if (t0.type == lexer::token::e_symbol)
               {
                  if (ignore_set_.end() != ignore_set_.find(t0.value))
                  {
                     return -1;
                  }
                  else if (!t0.value.empty() && ('$' == t0.value[0]))
                  {
                     return -1;
                  }
               }

               if (t1.type == lexer::token::e_symbol)
               {
                  if (ignore_set_.end() != ignore_set_.find(t1.value))
                  {
                     return -1;
                  }
               }
               if      ((t0.type == lexer::token::e_number     ) && (t1.type == lexer::token::e_symbol     )) match = true;
               else if ((t0.type == lexer::token::e_number     ) && (t1.type == lexer::token::e_lbracket   )) match = true;
               else if ((t0.type == lexer::token::e_number     ) && (t1.type == lexer::token::e_lcrlbracket)) match = true;
               else if ((t0.type == lexer::token::e_number     ) && (t1.type == lexer::token::e_lsqrbracket)) match = true;
               else if ((t0.type == lexer::token::e_symbol     ) && (t1.type == lexer::token::e_number     )) match = true;
               else if ((t0.type == lexer::token::e_rbracket   ) && (t1.type == lexer::token::e_number     )) match = true;
               else if ((t0.type == lexer::token::e_rcrlbracket) && (t1.type == lexer::token::e_number     )) match = true;
               else if ((t0.type == lexer::token::e_rsqrbracket) && (t1.type == lexer::token::e_number     )) match = true;
               else if ((t0.type == lexer::token::e_rbracket   ) && (t1.type == lexer::token::e_symbol     )) match = true;
               else if ((t0.type == lexer::token::e_rcrlbracket) && (t1.type == lexer::token::e_symbol     )) match = true;
               else if ((t0.type == lexer::token::e_rsqrbracket) && (t1.type == lexer::token::e_symbol     )) match = true;
               else if ((t0.type == lexer::token::e_symbol     ) && (t1.type == lexer::token::e_symbol     )) match = true;

               return (match) ? 1 : -1;
            }

         private:

            std::set<std::string,details::ilesscompare> ignore_set_;
         };

         class operator_joiner : public token_joiner
         {
         public:

            explicit operator_joiner(const std::size_t& stride)
            : token_joiner(stride)
            {}

            inline bool join(const lexer::token& t0, const lexer::token& t1, lexer::token& t)
            {
               // ': =' --> ':='
               if ((t0.type == lexer::token::e_colon) && (t1.type == lexer::token::e_eq))
               {
                  t.type     = lexer::token::e_assign;
                  t.value    = ":=";
                  t.position = t0.position;

                  return true;
               }
               // '+ =' --> '+='
               else if ((t0.type == lexer::token::e_add) && (t1.type == lexer::token::e_eq))
               {
                  t.type     = lexer::token::e_addass;
                  t.value    = "+=";
                  t.position = t0.position;

                  return true;
               }
               // '- =' --> '-='
               else if ((t0.type == lexer::token::e_sub) && (t1.type == lexer::token::e_eq))
               {
                  t.type     = lexer::token::e_subass;
                  t.value    = "-=";
                  t.position = t0.position;

                  return true;
               }
               // '* =' --> '*='
               else if ((t0.type == lexer::token::e_mul) && (t1.type == lexer::token::e_eq))
               {
                  t.type     = lexer::token::e_mulass;
                  t.value    = "*=";
                  t.position = t0.position;

                  return true;
               }
               // '/ =' --> '/='
               else if ((t0.type == lexer::token::e_div) && (t1.type == lexer::token::e_eq))
               {
                  t.type     = lexer::token::e_divass;
                  t.value    = "/=";
                  t.position = t0.position;

                  return true;
               }
               // '% =' --> '%='
               else if ((t0.type == lexer::token::e_mod) && (t1.type == lexer::token::e_eq))
               {
                  t.type     = lexer::token::e_modass;
                  t.value    = "%=";
                  t.position = t0.position;

                  return true;
               }
               // '> =' --> '>='
               else if ((t0.type == lexer::token::e_gt) && (t1.type == lexer::token::e_eq))
               {
                  t.type     = lexer::token::e_gte;
                  t.value    = ">=";
                  t.position = t0.position;

                  return true;
               }
               // '< =' --> '<='
               else if ((t0.type == lexer::token::e_lt) && (t1.type == lexer::token::e_eq))
               {
                  t.type     = lexer::token::e_lte;
                  t.value    = "<=";
                  t.position = t0.position;

                  return true;
               }
               // '= =' --> '=='
               else if ((t0.type == lexer::token::e_eq) && (t1.type == lexer::token::e_eq))
               {
                  t.type     = lexer::token::e_eq;
                  t.value    = "==";
                  t.position = t0.position;

                  return true;
               }
               // '! =' --> '!='
               else if ((static_cast<details::char_t>(t0.type) == '!') && (t1.type == lexer::token::e_eq))
               {
                  t.type     = lexer::token::e_ne;
                  t.value    = "!=";
                  t.position = t0.position;

                  return true;
               }
               // '< >' --> '<>'
               else if ((t0.type == lexer::token::e_lt) && (t1.type == lexer::token::e_gt))
               {
                  t.type     = lexer::token::e_ne;
                  t.value    = "<>";
                  t.position = t0.position;

                  return true;
               }
               // '<= >' --> '<=>'
               else if ((t0.type == lexer::token::e_lte) && (t1.type == lexer::token::e_gt))
               {
                  t.type     = lexer::token::e_swap;
                  t.value    = "<=>";
                  t.position = t0.position;

                  return true;
               }
               // '+ -' --> '-'
               else if ((t0.type == lexer::token::e_add) && (t1.type == lexer::token::e_sub))
               {
                  t.type     = lexer::token::e_sub;
                  t.value    = "-";
                  t.position = t0.position;

                  return true;
               }
               // '- +' --> '-'
               else if ((t0.type == lexer::token::e_sub) && (t1.type == lexer::token::e_add))
               {
                  t.type     = lexer::token::e_sub;
                  t.value    = "-";
                  t.position = t0.position;

                  return true;
               }
               // '- -' --> '+'
               else if ((t0.type == lexer::token::e_sub) && (t1.type == lexer::token::e_sub))
               {
                  /*
                     Note: May need to reconsider this when wanting to implement
                     pre/postfix decrement operator
                  */
                  t.type     = lexer::token::e_add;
                  t.value    = "+";
                  t.position = t0.position;

                  return true;
               }
               else
                  return false;
            }

            inline bool join(const lexer::token& t0,
                             const lexer::token& t1,
                             const lexer::token& t2,
                             lexer::token& t)
            {
               // '[ * ]' --> '[*]'
               if (
                    (t0.type == lexer::token::e_lsqrbracket) &&
                    (t1.type == lexer::token::e_mul        ) &&
                    (t2.type == lexer::token::e_rsqrbracket)
                  )
               {
                  t.type     = lexer::token::e_symbol;
                  t.value    = "[*]";
                  t.position = t0.position;

                  return true;
               }
               else
                  return false;
            }
         };

         class bracket_checker : public lexer::token_scanner
         {
         public:

            using lexer::token_scanner::operator();

            bracket_checker()
            : token_scanner(1)
            , state_(true)
            {}

            bool result()
            {
               if (!stack_.empty())
               {
                  lexer::token t;
                  t.value      = stack_.top().first;
                  t.position   = stack_.top().second;
                  error_token_ = t;
                  state_       = false;

                  return false;
               }
               else
                  return state_;
            }

            lexer::token error_token()
            {
               return error_token_;
            }

            void reset()
            {
               // Why? because msvc doesn't support swap properly.
               stack_ = std::stack<std::pair<char,std::size_t> >();
               state_ = true;
               error_token_.clear();
            }

            bool operator() (const lexer::token& t)
            {
               if (
                    !t.value.empty()                       &&
                    (lexer::token::e_string != t.type)     &&
                    (lexer::token::e_symbol != t.type)     &&
                    exprtk::details::is_bracket(t.value[0])
                  )
               {
                  details::char_t c = t.value[0];

                  if      (t.type == lexer::token::e_lbracket   ) stack_.push(std::make_pair(')',t.position));
                  else if (t.type == lexer::token::e_lcrlbracket) stack_.push(std::make_pair('}',t.position));
                  else if (t.type == lexer::token::e_lsqrbracket) stack_.push(std::make_pair(']',t.position));
                  else if (exprtk::details::is_right_bracket(c))
                  {
                     if (stack_.empty())
                     {
                        state_       = false;
                        error_token_ = t;

                        return false;
                     }
                     else if (c != stack_.top().first)
                     {
                        state_       = false;
                        error_token_ = t;

                        return false;
                     }
                     else
                        stack_.pop();
                  }
               }

               return true;
            }

         private:

            bool state_;
            std::stack<std::pair<char,std::size_t> > stack_;
            lexer::token error_token_;
         };

         class numeric_checker : public lexer::token_scanner
         {
         public:

            using lexer::token_scanner::operator();

            numeric_checker()
            : token_scanner (1)
            , current_index_(0)
            {}

            bool result()
            {
               return error_list_.empty();
            }

            void reset()
            {
               error_list_.clear();
               current_index_ = 0;
            }

            bool operator() (const lexer::token& t)
            {
               if (token::e_number == t.type)
               {
                  double v;

                  if (!exprtk::details::string_to_real(t.value,v))
                  {
                     error_list_.push_back(current_index_);
                  }
               }

               ++current_index_;

               return true;
            }

            std::size_t error_count() const
            {
               return error_list_.size();
            }

            std::size_t error_index(const std::size_t& i)
            {
               if (i < error_list_.size())
                  return error_list_[i];
               else
                  return std::numeric_limits<std::size_t>::max();
            }

            void clear_errors()
            {
               error_list_.clear();
            }

         private:

            std::size_t current_index_;
            std::vector<std::size_t> error_list_;
         };

         class symbol_replacer : public lexer::token_modifier
         {
         private:

            typedef std::map<std::string,std::pair<std::string,token::token_type>,details::ilesscompare> replace_map_t;

         public:

            bool remove(const std::string& target_symbol)
            {
               const replace_map_t::iterator itr = replace_map_.find(target_symbol);

               if (replace_map_.end() == itr)
                  return false;

               replace_map_.erase(itr);

               return true;
            }

            bool add_replace(const std::string& target_symbol,
                             const std::string& replace_symbol,
                             const lexer::token::token_type token_type = lexer::token::e_symbol)
            {
               const replace_map_t::iterator itr = replace_map_.find(target_symbol);

               if (replace_map_.end() != itr)
               {
                  return false;
               }

               replace_map_[target_symbol] = std::make_pair(replace_symbol,token_type);

               return true;
            }

            void clear()
            {
               replace_map_.clear();
            }

         private:

            bool modify(lexer::token& t)
            {
               if (lexer::token::e_symbol == t.type)
               {
                  if (replace_map_.empty())
                     return false;

                  const replace_map_t::iterator itr = replace_map_.find(t.value);

                  if (replace_map_.end() != itr)
                  {
                     t.value = itr->second.first;
                     t.type  = itr->second.second;

                     return true;
                  }
               }

               return false;
            }

            replace_map_t replace_map_;
         };

         class sequence_validator : public lexer::token_scanner
         {
         private:

            typedef std::pair<lexer::token::token_type,lexer::token::token_type> token_pair_t;
            typedef std::set<token_pair_t> set_t;

         public:

            using lexer::token_scanner::operator();

            sequence_validator()
            : lexer::token_scanner(2)
            {
               add_invalid(lexer::token::e_number, lexer::token::e_number);
               add_invalid(lexer::token::e_string, lexer::token::e_string);
               add_invalid(lexer::token::e_number, lexer::token::e_string);
               add_invalid(lexer::token::e_string, lexer::token::e_number);

               add_invalid_set1(lexer::token::e_assign );
               add_invalid_set1(lexer::token::e_shr    );
               add_invalid_set1(lexer::token::e_shl    );
               add_invalid_set1(lexer::token::e_lte    );
               add_invalid_set1(lexer::token::e_ne     );
               add_invalid_set1(lexer::token::e_gte    );
               add_invalid_set1(lexer::token::e_lt     );
               add_invalid_set1(lexer::token::e_gt     );
               add_invalid_set1(lexer::token::e_eq     );
               add_invalid_set1(lexer::token::e_comma  );
               add_invalid_set1(lexer::token::e_add    );
               add_invalid_set1(lexer::token::e_sub    );
               add_invalid_set1(lexer::token::e_div    );
               add_invalid_set1(lexer::token::e_mul    );
               add_invalid_set1(lexer::token::e_mod    );
               add_invalid_set1(lexer::token::e_pow    );
               add_invalid_set1(lexer::token::e_colon  );
               add_invalid_set1(lexer::token::e_ternary);
            }

            bool result()
            {
               return error_list_.empty();
            }

            bool operator() (const lexer::token& t0, const lexer::token& t1)
            {
               const set_t::value_type p = std::make_pair(t0.type,t1.type);

               if (invalid_bracket_check(t0.type,t1.type))
               {
                  error_list_.push_back(std::make_pair(t0,t1));
               }
               else if (invalid_comb_.find(p) != invalid_comb_.end())
               {
                  error_list_.push_back(std::make_pair(t0,t1));
               }

               return true;
            }

            std::size_t error_count() const
            {
               return error_list_.size();
            }

            std::pair<lexer::token,lexer::token> error(const std::size_t index)
            {
               if (index < error_list_.size())
               {
                  return error_list_[index];
               }
               else
               {
                  static const lexer::token error_token;
                  return std::make_pair(error_token,error_token);
               }
            }

            void clear_errors()
            {
               error_list_.clear();
            }

         private:

            void add_invalid(const lexer::token::token_type base, const lexer::token::token_type t)
            {
               invalid_comb_.insert(std::make_pair(base,t));
            }

            void add_invalid_set1(const lexer::token::token_type t)
            {
               add_invalid(t, lexer::token::e_assign);
               add_invalid(t, lexer::token::e_shr   );
               add_invalid(t, lexer::token::e_shl   );
               add_invalid(t, lexer::token::e_lte   );
               add_invalid(t, lexer::token::e_ne    );
               add_invalid(t, lexer::token::e_gte   );
               add_invalid(t, lexer::token::e_lt    );
               add_invalid(t, lexer::token::e_gt    );
               add_invalid(t, lexer::token::e_eq    );
               add_invalid(t, lexer::token::e_comma );
               add_invalid(t, lexer::token::e_div   );
               add_invalid(t, lexer::token::e_mul   );
               add_invalid(t, lexer::token::e_mod   );
               add_invalid(t, lexer::token::e_pow   );
               add_invalid(t, lexer::token::e_colon );
            }

            bool invalid_bracket_check(const lexer::token::token_type base, const lexer::token::token_type t)
            {
               if (details::is_right_bracket(static_cast<details::char_t>(base)))
               {
                  switch (t)
                  {
                     case lexer::token::e_assign : return (']' != base);
                     case lexer::token::e_string : return (')' != base);
                     default                     : return false;
                  }
               }
               else if (details::is_left_bracket(static_cast<details::char_t>(base)))
               {
                  if (details::is_right_bracket(static_cast<details::char_t>(t)))
                     return false;
                  else if (details::is_left_bracket(static_cast<details::char_t>(t)))
                     return false;
                  else
                  {
                     switch (t)
                     {
                        case lexer::token::e_number  : return false;
                        case lexer::token::e_symbol  : return false;
                        case lexer::token::e_string  : return false;
                        case lexer::token::e_add     : return false;
                        case lexer::token::e_sub     : return false;
                        case lexer::token::e_colon   : return false;
                        case lexer::token::e_ternary : return false;
                        default                      : return true ;
                     }
                  }
               }
               else if (details::is_right_bracket(static_cast<details::char_t>(t)))
               {
                  switch (base)
                  {
                     case lexer::token::e_number  : return false;
                     case lexer::token::e_symbol  : return false;
                     case lexer::token::e_string  : return false;
                     case lexer::token::e_eof     : return false;
                     case lexer::token::e_colon   : return false;
                     case lexer::token::e_ternary : return false;
                     default                      : return true ;
                  }
               }
               else if (details::is_left_bracket(static_cast<details::char_t>(t)))
               {
                  switch (base)
                  {
                     case lexer::token::e_rbracket    : return true;
                     case lexer::token::e_rsqrbracket : return true;
                     case lexer::token::e_rcrlbracket : return true;
                     default                          : return false;
                  }
               }

               return false;
            }

            set_t invalid_comb_;
            std::vector<std::pair<lexer::token,lexer::token> > error_list_;
         };

         class sequence_validator_3tokens : public lexer::token_scanner
         {
         private:

            typedef lexer::token::token_type token_t;
            typedef std::pair<token_t,std::pair<token_t,token_t> > token_triplet_t;
            typedef std::set<token_triplet_t> set_t;

         public:

            using lexer::token_scanner::operator();

            sequence_validator_3tokens()
            : lexer::token_scanner(3)
            {
               add_invalid(lexer::token::e_number , lexer::token::e_number , lexer::token::e_number);
               add_invalid(lexer::token::e_string , lexer::token::e_string , lexer::token::e_string);
               add_invalid(lexer::token::e_comma  , lexer::token::e_comma  , lexer::token::e_comma );

               add_invalid(lexer::token::e_add    , lexer::token::e_add    , lexer::token::e_add   );
               add_invalid(lexer::token::e_sub    , lexer::token::e_sub    , lexer::token::e_sub   );
               add_invalid(lexer::token::e_div    , lexer::token::e_div    , lexer::token::e_div   );
               add_invalid(lexer::token::e_mul    , lexer::token::e_mul    , lexer::token::e_mul   );
               add_invalid(lexer::token::e_mod    , lexer::token::e_mod    , lexer::token::e_mod   );
               add_invalid(lexer::token::e_pow    , lexer::token::e_pow    , lexer::token::e_pow   );

               add_invalid(lexer::token::e_add    , lexer::token::e_sub    , lexer::token::e_add   );
               add_invalid(lexer::token::e_sub    , lexer::token::e_add    , lexer::token::e_sub   );
               add_invalid(lexer::token::e_div    , lexer::token::e_mul    , lexer::token::e_div   );
               add_invalid(lexer::token::e_mul    , lexer::token::e_div    , lexer::token::e_mul   );
               add_invalid(lexer::token::e_mod    , lexer::token::e_pow    , lexer::token::e_mod   );
               add_invalid(lexer::token::e_pow    , lexer::token::e_mod    , lexer::token::e_pow   );
            }

            bool result()
            {
               return error_list_.empty();
            }

            bool operator() (const lexer::token& t0, const lexer::token& t1, const lexer::token& t2)
            {
               const set_t::value_type p = std::make_pair(t0.type,std::make_pair(t1.type,t2.type));

               if (invalid_comb_.find(p) != invalid_comb_.end())
               {
                  error_list_.push_back(std::make_pair(t0,t1));
               }

               return true;
            }

            std::size_t error_count() const
            {
               return error_list_.size();
            }

            std::pair<lexer::token,lexer::token> error(const std::size_t index)
            {
               if (index < error_list_.size())
               {
                  return error_list_[index];
               }
               else
               {
                  static const lexer::token error_token;
                  return std::make_pair(error_token,error_token);
               }
            }

            void clear_errors()
            {
               error_list_.clear();
            }

         private:

            void add_invalid(const token_t t0, const token_t t1, const token_t t2)
            {
               invalid_comb_.insert(std::make_pair(t0,std::make_pair(t1,t2)));
            }

            set_t invalid_comb_;
            std::vector<std::pair<lexer::token,lexer::token> > error_list_;
         };

         struct helper_assembly
         {
            inline bool register_scanner(lexer::token_scanner* scanner)
            {
               if (token_scanner_list.end() != std::find(token_scanner_list.begin(),
                                                         token_scanner_list.end  (),
                                                         scanner))
               {
                  return false;
               }

               token_scanner_list.push_back(scanner);

               return true;
            }

            inline bool register_modifier(lexer::token_modifier* modifier)
            {
               if (token_modifier_list.end() != std::find(token_modifier_list.begin(),
                                                          token_modifier_list.end  (),
                                                          modifier))
               {
                  return false;
               }

               token_modifier_list.push_back(modifier);

               return true;
            }

            inline bool register_joiner(lexer::token_joiner* joiner)
            {
               if (token_joiner_list.end() != std::find(token_joiner_list.begin(),
                                                        token_joiner_list.end  (),
                                                        joiner))
               {
                  return false;
               }

               token_joiner_list.push_back(joiner);

               return true;
            }

            inline bool register_inserter(lexer::token_inserter* inserter)
            {
               if (token_inserter_list.end() != std::find(token_inserter_list.begin(),
                                                          token_inserter_list.end  (),
                                                          inserter))
               {
                  return false;
               }

               token_inserter_list.push_back(inserter);

               return true;
            }

            inline bool run_modifiers(lexer::generator& g)
            {
               error_token_modifier = reinterpret_cast<lexer::token_modifier*>(0);

               for (std::size_t i = 0; i < token_modifier_list.size(); ++i)
               {
                  lexer::token_modifier& modifier = (*token_modifier_list[i]);

                  modifier.reset();
                  modifier.process(g);

                  if (!modifier.result())
                  {
                     error_token_modifier = token_modifier_list[i];

                     return false;
                  }
               }

               return true;
            }

            inline bool run_joiners(lexer::generator& g)
            {
               error_token_joiner = reinterpret_cast<lexer::token_joiner*>(0);

               for (std::size_t i = 0; i < token_joiner_list.size(); ++i)
               {
                  lexer::token_joiner& joiner = (*token_joiner_list[i]);

                  joiner.reset();
                  joiner.process(g);

                  if (!joiner.result())
                  {
                     error_token_joiner = token_joiner_list[i];

                     return false;
                  }
               }

               return true;
            }

            inline bool run_inserters(lexer::generator& g)
            {
               error_token_inserter = reinterpret_cast<lexer::token_inserter*>(0);

               for (std::size_t i = 0; i < token_inserter_list.size(); ++i)
               {
                  lexer::token_inserter& inserter = (*token_inserter_list[i]);

                  inserter.reset();
                  inserter.process(g);

                  if (!inserter.result())
                  {
                     error_token_inserter = token_inserter_list[i];

                     return false;
                  }
               }

               return true;
            }

            inline bool run_scanners(lexer::generator& g)
            {
               error_token_scanner = reinterpret_cast<lexer::token_scanner*>(0);

               for (std::size_t i = 0; i < token_scanner_list.size(); ++i)
               {
                  lexer::token_scanner& scanner = (*token_scanner_list[i]);

                  scanner.reset();
                  scanner.process(g);

                  if (!scanner.result())
                  {
                     error_token_scanner = token_scanner_list[i];

                     return false;
                  }
               }

               return true;
            }

            std::vector<lexer::token_scanner*>  token_scanner_list;
            std::vector<lexer::token_modifier*> token_modifier_list;
            std::vector<lexer::token_joiner*>   token_joiner_list;
            std::vector<lexer::token_inserter*> token_inserter_list;

            lexer::token_scanner*  error_token_scanner;
            lexer::token_modifier* error_token_modifier;
            lexer::token_joiner*   error_token_joiner;
            lexer::token_inserter* error_token_inserter;
         };
      }

      class parser_helper
      {
      public:

         typedef token     token_t;
         typedef generator generator_t;

         inline bool init(const std::string& str)
         {
            if (!lexer_.process(str))
            {
               return false;
            }

            lexer_.begin();

            next_token();

            return true;
         }

         inline generator_t& lexer()
         {
            return lexer_;
         }

         inline const generator_t& lexer() const
         {
            return lexer_;
         }

         inline void store_token()
         {
            lexer_.store();
            store_current_token_ = current_token_;
         }

         inline void restore_token()
         {
            lexer_.restore();
            current_token_ = store_current_token_;
         }

         inline void next_token()
         {
            current_token_ = lexer_.next_token();
         }

         inline const token_t& current_token() const
         {
            return current_token_;
         }

         enum token_advance_mode
         {
            e_hold    = 0,
            e_advance = 1
         };

         inline void advance_token(const token_advance_mode mode)
         {
            if (e_advance == mode)
            {
               next_token();
            }
         }

         inline bool token_is(const token_t::token_type& ttype, const token_advance_mode mode = e_advance)
         {
            if (current_token().type != ttype)
            {
               return false;
            }

            advance_token(mode);

            return true;
         }

         inline bool token_is(const token_t::token_type& ttype,
                              const std::string& value,
                              const token_advance_mode mode = e_advance)
         {
            if (
                 (current_token().type != ttype) ||
                 !exprtk::details::imatch(value,current_token().value)
               )
            {
               return false;
            }

            advance_token(mode);

            return true;
         }

         inline bool peek_token_is(const token_t::token_type& ttype)
         {
            return (lexer_.peek_next_token().type == ttype);
         }

         inline bool peek_token_is(const std::string& s)
         {
            return (exprtk::details::imatch(lexer_.peek_next_token().value,s));
         }

      private:

         generator_t lexer_;
         token_t     current_token_;
         token_t     store_current_token_;
      };
   }

   template <typename T>
   class vector_view
   {
   public:

      typedef T* data_ptr_t;

      vector_view(data_ptr_t data, const std::size_t& size)
      : size_(size)
      , data_(data)
      , data_ref_(0)
      {}

      vector_view(const vector_view<T>& vv)
      : size_(vv.size_)
      , data_(vv.data_)
      , data_ref_(0)
      {}

      inline void rebase(data_ptr_t data)
      {
         data_ = data;

         if (!data_ref_.empty())
         {
            for (std::size_t i = 0; i < data_ref_.size(); ++i)
            {
               (*data_ref_[i]) = data;
            }
         }
      }

      inline data_ptr_t data() const
      {
         return data_;
      }

      inline std::size_t size() const
      {
         return size_;
      }

      inline const T& operator[](const std::size_t index) const
      {
         return data_[index];
      }

      inline T& operator[](const std::size_t index)
      {
         return data_[index];
      }

      void set_ref(data_ptr_t* data_ref)
      {
         data_ref_.push_back(data_ref);
      }

   private:

      const std::size_t size_;
      data_ptr_t  data_;
      std::vector<data_ptr_t*> data_ref_;
   };

   template <typename T>
   inline vector_view<T> make_vector_view(T* data,
                                          const std::size_t size, const std::size_t offset = 0)
   {
      return vector_view<T>(data + offset, size);
   }

   template <typename T>
   inline vector_view<T> make_vector_view(std::vector<T>& v,
                                          const std::size_t size, const std::size_t offset = 0)
   {
      return vector_view<T>(v.data() + offset, size);
   }

   template <typename T> class results_context;

   template <typename T>
   struct type_store
   {
      enum store_type
      {
         e_unknown,
         e_scalar ,
         e_vector ,
         e_string
      };

      type_store()
      : data(0)
      , size(0)
      , type(e_unknown)
      {}

      union
      {
         void* data;
         T*    vec_data;
      };

      std::size_t size;
      store_type  type;

      class parameter_list
      {
      public:

         explicit parameter_list(std::vector<type_store>& pl)
         : parameter_list_(pl)
         {}

         inline bool empty() const
         {
            return parameter_list_.empty();
         }

         inline std::size_t size() const
         {
            return parameter_list_.size();
         }

         inline type_store& operator[](const std::size_t& index)
         {
            return parameter_list_[index];
         }

         inline const type_store& operator[](const std::size_t& index) const
         {
            return parameter_list_[index];
         }

         inline type_store& front()
         {
            return parameter_list_[0];
         }

         inline const type_store& front() const
         {
            return parameter_list_[0];
         }

         inline type_store& back()
         {
            return parameter_list_.back();
         }

         inline const type_store& back() const
         {
            return parameter_list_.back();
         }

      private:

         std::vector<type_store>& parameter_list_;

         friend class results_context<T>;
      };

      template <typename ViewType>
      struct type_view
      {
         typedef type_store<T> type_store_t;
         typedef ViewType      value_t;

         explicit type_view(type_store_t& ts)
         : ts_(ts)
         , data_(reinterpret_cast<value_t*>(ts_.data))
         {}

         explicit type_view(const type_store_t& ts)
         : ts_(const_cast<type_store_t&>(ts))
         , data_(reinterpret_cast<value_t*>(ts_.data))
         {}

         inline std::size_t size() const
         {
            return ts_.size;
         }

         inline value_t& operator[](const std::size_t& i)
         {
            return data_[i];
         }

         inline const value_t& operator[](const std::size_t& i) const
         {
            return data_[i];
         }

         inline const value_t* begin() const { return data_; }
         inline       value_t* begin()       { return data_; }

         inline const value_t* end() const
         {
            return static_cast<value_t*>(data_ + ts_.size);
         }

         inline value_t* end()
         {
            return static_cast<value_t*>(data_ + ts_.size);
         }

         type_store_t& ts_;
         value_t* data_;
      };

      typedef type_view<T>    vector_view;
      typedef type_view<char> string_view;

      struct scalar_view
      {
         typedef type_store<T> type_store_t;
         typedef T value_t;

         explicit scalar_view(type_store_t& ts)
         : v_(*reinterpret_cast<value_t*>(ts.data))
         {}

         explicit scalar_view(const type_store_t& ts)
         : v_(*reinterpret_cast<value_t*>(const_cast<type_store_t&>(ts).data))
         {}

         inline value_t& operator() ()
         {
            return v_;
         }

         inline const value_t& operator() () const
         {
            return v_;
         }

         template <typename IntType>
         inline bool to_int(IntType& i) const
         {
            if (!exprtk::details::numeric::is_integer(v_))
               return false;

            i = static_cast<IntType>(v_);

            return true;
         }

         template <typename UIntType>
         inline bool to_uint(UIntType& u) const
         {
            if (v_ < T(0))
               return false;
            else if (!exprtk::details::numeric::is_integer(v_))
               return false;

            u = static_cast<UIntType>(v_);

            return true;
         }

         T& v_;
      };
   };

   template <typename StringView>
   inline std::string to_str(const StringView& view)
   {
      return std::string(view.begin(),view.size());
   }

   template <typename T>
   class results_context
   {
   public:

      typedef type_store<T> type_store_t;

      results_context()
      : results_available_(false)
      {}

      inline std::size_t count() const
      {
         if (results_available_)
            return parameter_list_.size();
         else
            return 0;
      }

      inline type_store_t& operator[](const std::size_t& index)
      {
         return parameter_list_[index];
      }

      inline const type_store_t& operator[](const std::size_t& index) const
      {
         return parameter_list_[index];
      }

   private:

      inline void clear()
      {
         results_available_ = false;
      }

      typedef std::vector<type_store_t> ts_list_t;
      typedef typename type_store_t::parameter_list parameter_list_t;

      inline void assign(const parameter_list_t& pl)
      {
         parameter_list_    = pl.parameter_list_;
         results_available_ = true;
      }

      bool results_available_;
      ts_list_t parameter_list_;

   };

   namespace details
   {
      enum operator_type
      {
         e_default , e_null    , e_add     , e_sub     ,
         e_mul     , e_div     , e_mod     , e_pow     ,
         e_atan2   , e_min     , e_max     , e_avg     ,
         e_sum     , e_prod    , e_lt      , e_lte     ,
         e_eq      , e_equal   , e_ne      , e_nequal  ,
         e_gte     , e_gt      , e_and     , e_nand    ,
         e_or      , e_nor     , e_xor     , e_xnor    ,
         e_mand    , e_mor     , e_scand   , e_scor    ,
         e_shr     , e_shl     , e_abs     , e_acos    ,
         e_acosh   , e_asin    , e_asinh   , e_atan    ,
         e_atanh   , e_ceil    , e_cos     , e_cosh    ,
         e_exp     , e_expm1   , e_floor   , e_log     ,
         e_log10   , e_log2    , e_log1p   , e_logn    ,
         e_neg     , e_pos     , e_round   , e_roundn  ,
         e_root    , e_sqrt    , e_sin     , e_sinc    ,
         e_sinh    , e_sec     , e_csc     , e_tan     ,
         e_tanh    , e_cot     , e_clamp   , e_iclamp  ,
         e_inrange , e_sgn     , e_r2d     , e_d2r     ,
         e_d2g     , e_g2d     , e_hypot   , e_notl    ,
         e_erf     , e_erfc    , e_ncdf    , e_frac    ,
         e_trunc   , e_assign  , e_addass  , e_subass  ,
         e_mulass  , e_divass  , e_modass  , e_in      ,
         e_like    , e_ilike   , e_multi   , e_smulti  ,
         e_swap    ,

         // Do not add new functions/operators after this point.
         e_sf00 = 1000, e_sf01 = 1001, e_sf02 = 1002, e_sf03 = 1003,
         e_sf04 = 1004, e_sf05 = 1005, e_sf06 = 1006, e_sf07 = 1007,
         e_sf08 = 1008, e_sf09 = 1009, e_sf10 = 1010, e_sf11 = 1011,
         e_sf12 = 1012, e_sf13 = 1013, e_sf14 = 1014, e_sf15 = 1015,
         e_sf16 = 1016, e_sf17 = 1017, e_sf18 = 1018, e_sf19 = 1019,
         e_sf20 = 1020, e_sf21 = 1021, e_sf22 = 1022, e_sf23 = 1023,
         e_sf24 = 1024, e_sf25 = 1025, e_sf26 = 1026, e_sf27 = 1027,
         e_sf28 = 1028, e_sf29 = 1029, e_sf30 = 1030, e_sf31 = 1031,
         e_sf32 = 1032, e_sf33 = 1033, e_sf34 = 1034, e_sf35 = 1035,
         e_sf36 = 1036, e_sf37 = 1037, e_sf38 = 1038, e_sf39 = 1039,
         e_sf40 = 1040, e_sf41 = 1041, e_sf42 = 1042, e_sf43 = 1043,
         e_sf44 = 1044, e_sf45 = 1045, e_sf46 = 1046, e_sf47 = 1047,
         e_sf48 = 1048, e_sf49 = 1049, e_sf50 = 1050, e_sf51 = 1051,
         e_sf52 = 1052, e_sf53 = 1053, e_sf54 = 1054, e_sf55 = 1055,
         e_sf56 = 1056, e_sf57 = 1057, e_sf58 = 1058, e_sf59 = 1059,
         e_sf60 = 1060, e_sf61 = 1061, e_sf62 = 1062, e_sf63 = 1063,
         e_sf64 = 1064, e_sf65 = 1065, e_sf66 = 1066, e_sf67 = 1067,
         e_sf68 = 1068, e_sf69 = 1069, e_sf70 = 1070, e_sf71 = 1071,
         e_sf72 = 1072, e_sf73 = 1073, e_sf74 = 1074, e_sf75 = 1075,
         e_sf76 = 1076, e_sf77 = 1077, e_sf78 = 1078, e_sf79 = 1079,
         e_sf80 = 1080, e_sf81 = 1081, e_sf82 = 1082, e_sf83 = 1083,
         e_sf84 = 1084, e_sf85 = 1085, e_sf86 = 1086, e_sf87 = 1087,
         e_sf88 = 1088, e_sf89 = 1089, e_sf90 = 1090, e_sf91 = 1091,
         e_sf92 = 1092, e_sf93 = 1093, e_sf94 = 1094, e_sf95 = 1095,
         e_sf96 = 1096, e_sf97 = 1097, e_sf98 = 1098, e_sf99 = 1099,
         e_sffinal  = 1100,
         e_sf4ext00 = 2000, e_sf4ext01 = 2001, e_sf4ext02 = 2002, e_sf4ext03 = 2003,
         e_sf4ext04 = 2004, e_sf4ext05 = 2005, e_sf4ext06 = 2006, e_sf4ext07 = 2007,
         e_sf4ext08 = 2008, e_sf4ext09 = 2009, e_sf4ext10 = 2010, e_sf4ext11 = 2011,
         e_sf4ext12 = 2012, e_sf4ext13 = 2013, e_sf4ext14 = 2014, e_sf4ext15 = 2015,
         e_sf4ext16 = 2016, e_sf4ext17 = 2017, e_sf4ext18 = 2018, e_sf4ext19 = 2019,
         e_sf4ext20 = 2020, e_sf4ext21 = 2021, e_sf4ext22 = 2022, e_sf4ext23 = 2023,
         e_sf4ext24 = 2024, e_sf4ext25 = 2025, e_sf4ext26 = 2026, e_sf4ext27 = 2027,
         e_sf4ext28 = 2028, e_sf4ext29 = 2029, e_sf4ext30 = 2030, e_sf4ext31 = 2031,
         e_sf4ext32 = 2032, e_sf4ext33 = 2033, e_sf4ext34 = 2034, e_sf4ext35 = 2035,
         e_sf4ext36 = 2036, e_sf4ext37 = 2037, e_sf4ext38 = 2038, e_sf4ext39 = 2039,
         e_sf4ext40 = 2040, e_sf4ext41 = 2041, e_sf4ext42 = 2042, e_sf4ext43 = 2043,
         e_sf4ext44 = 2044, e_sf4ext45 = 2045, e_sf4ext46 = 2046, e_sf4ext47 = 2047,
         e_sf4ext48 = 2048, e_sf4ext49 = 2049, e_sf4ext50 = 2050, e_sf4ext51 = 2051,
         e_sf4ext52 = 2052, e_sf4ext53 = 2053, e_sf4ext54 = 2054, e_sf4ext55 = 2055,
         e_sf4ext56 = 2056, e_sf4ext57 = 2057, e_sf4ext58 = 2058, e_sf4ext59 = 2059,
         e_sf4ext60 = 2060, e_sf4ext61 = 2061
      };

      inline std::string to_str(const operator_type opr)
      {
         switch (opr)
         {
            case e_add    : return  "+"  ;
            case e_sub    : return  "-"  ;
            case e_mul    : return  "*"  ;
            case e_div    : return  "/"  ;
            case e_mod    : return  "%"  ;
            case e_pow    : return  "^"  ;
            case e_assign : return ":="  ;
            case e_addass : return "+="  ;
            case e_subass : return "-="  ;
            case e_mulass : return "*="  ;
            case e_divass : return "/="  ;
            case e_modass : return "%="  ;
            case e_lt     : return  "<"  ;
            case e_lte    : return "<="  ;
            case e_eq     : return "=="  ;
            case e_equal  : return  "="  ;
            case e_ne     : return "!="  ;
            case e_nequal : return "<>"  ;
            case e_gte    : return ">="  ;
            case e_gt     : return  ">"  ;
            case e_and    : return "and" ;
            case e_or     : return "or"  ;
            case e_xor    : return "xor" ;
            case e_nand   : return "nand";
            case e_nor    : return "nor" ;
            case e_xnor   : return "xnor";
            default       : return "N/A" ;
         }
      }

      struct base_operation_t
      {
         base_operation_t(const operator_type t, const unsigned int& np)
         : type(t)
         , num_params(np)
         {}

         operator_type type;
         unsigned int num_params;
      };

      namespace loop_unroll
      {
         const unsigned int global_loop_batch_size = 4;

         struct details
         {
            explicit details(const std::size_t& vsize,
                             const unsigned int loop_batch_size = global_loop_batch_size)
            : batch_size(loop_batch_size   )
            , remainder (vsize % batch_size)
            , upper_bound(static_cast<int>(vsize - (remainder ? loop_batch_size : 0)))
            {}

            unsigned int batch_size;
            int remainder;
            int upper_bound;
         };
      }

      #ifdef exprtk_enable_debugging
      inline void dump_ptr(const std::string& s, const void* ptr, const std::size_t size = 0)
      {
         if (size)
            exprtk_debug(("%s - addr: %p\n",s.c_str(),ptr));
         else
            exprtk_debug(("%s - addr: %p size: %d\n",
                          s.c_str(),
                          ptr,
                          static_cast<unsigned int>(size)));
      }
      #else
      inline void dump_ptr(const std::string&, const void*) {}
      inline void dump_ptr(const std::string&, const void*, const std::size_t) {}
      #endif

      template <typename T>
      class vec_data_store
      {
      public:

         typedef vec_data_store<T> type;
         typedef T* data_t;

      private:

         struct control_block
         {
            control_block()
            : ref_count(1)
            , size     (0)
            , data     (0)
            , destruct (true)
            {}

            explicit control_block(const std::size_t& dsize)
            : ref_count(1    )
            , size     (dsize)
            , data     (0    )
            , destruct (true )
            { create_data(); }

            control_block(const std::size_t& dsize, data_t dptr, bool dstrct = false)
            : ref_count(1     )
            , size     (dsize )
            , data     (dptr  )
            , destruct (dstrct)
            {}

           ~control_block()
            {
               if (data && destruct && (0 == ref_count))
               {
                  dump_ptr("~control_block() data",data);
                  delete[] data;
                  data = reinterpret_cast<data_t>(0);
               }
            }

            static inline control_block* create(const std::size_t& dsize, data_t data_ptr = data_t(0), bool dstrct = false)
            {
               if (dsize)
               {
                  if (0 == data_ptr)
                     return (new control_block(dsize));
                  else
                     return (new control_block(dsize, data_ptr, dstrct));
               }
               else
                  return (new control_block);
            }

            static inline void destroy(control_block*& cntrl_blck)
            {
               if (cntrl_blck)
               {
                  if (
                       (0 !=   cntrl_blck->ref_count) &&
                       (0 == --cntrl_blck->ref_count)
                     )
                  {
                     delete cntrl_blck;
                  }

                  cntrl_blck = 0;
               }
            }

            std::size_t ref_count;
            std::size_t size;
            data_t      data;
            bool        destruct;

         private:

            control_block(const control_block&) exprtk_delete;
            control_block& operator=(const control_block&) exprtk_delete;

            inline void create_data()
            {
               destruct = true;
               data     = new T[size];
               std::fill_n(data, size, T(0));
               dump_ptr("control_block::create_data() - data", data, size);
            }
         };

      public:

         vec_data_store()
         : control_block_(control_block::create(0))
         {}

         explicit vec_data_store(const std::size_t& size)
         : control_block_(control_block::create(size,reinterpret_cast<data_t>(0),true))
         {}

         vec_data_store(const std::size_t& size, data_t data, bool dstrct = false)
         : control_block_(control_block::create(size, data, dstrct))
         {}

         vec_data_store(const type& vds)
         {
            control_block_ = vds.control_block_;
            control_block_->ref_count++;
         }

        ~vec_data_store()
         {
            control_block::destroy(control_block_);
         }

         type& operator=(const type& vds)
         {
            if (this != &vds)
            {
               std::size_t final_size = min_size(control_block_, vds.control_block_);

               vds.control_block_->size = final_size;
                   control_block_->size = final_size;

               if (control_block_->destruct || (0 == control_block_->data))
               {
                  control_block::destroy(control_block_);

                  control_block_ = vds.control_block_;
                  control_block_->ref_count++;
               }
            }

            return (*this);
         }

         inline data_t data()
         {
            return control_block_->data;
         }

         inline data_t data() const
         {
            return control_block_->data;
         }

         inline std::size_t size() const
         {
            return control_block_->size;
         }

         inline data_t& ref()
         {
            return control_block_->data;
         }

         inline void dump() const
         {
            #ifdef exprtk_enable_debugging
            exprtk_debug(("size: %d\taddress:%p\tdestruct:%c\n",
                          size(),
                          data(),
                          (control_block_->destruct ? 'T' : 'F')));

            for (std::size_t i = 0; i < size(); ++i)
            {
               if (5 == i)
                  exprtk_debug(("\n"));

               exprtk_debug(("%15.10f ",data()[i]));
            }
            exprtk_debug(("\n"));
            #endif
         }

         static inline void match_sizes(type& vds0, type& vds1)
         {
            const std::size_t size = min_size(vds0.control_block_,vds1.control_block_);
            vds0.control_block_->size = size;
            vds1.control_block_->size = size;
         }

      private:

         static inline std::size_t min_size(const control_block* cb0, const control_block* cb1)
         {
            const std::size_t size0 = cb0->size;
            const std::size_t size1 = cb1->size;

            if (size0 && size1)
               return std::min(size0,size1);
            else
               return (size0) ? size0 : size1;
         }

         control_block* control_block_;
      };

      namespace numeric
      {
         namespace details
         {
            template <typename T>
            inline T process_impl(const operator_type operation, const T arg)
            {
               switch (operation)
               {
                  case e_abs   : return numeric::abs  (arg);
                  case e_acos  : return numeric::acos (arg);
                  case e_acosh : return numeric::acosh(arg);
                  case e_asin  : return numeric::asin (arg);
                  case e_asinh : return numeric::asinh(arg);
                  case e_atan  : return numeric::atan (arg);
                  case e_atanh : return numeric::atanh(arg);
                  case e_ceil  : return numeric::ceil (arg);
                  case e_cos   : return numeric::cos  (arg);
                  case e_cosh  : return numeric::cosh (arg);
                  case e_exp   : return numeric::exp  (arg);
                  case e_expm1 : return numeric::expm1(arg);
                  case e_floor : return numeric::floor(arg);
                  case e_log   : return numeric::log  (arg);
                  case e_log10 : return numeric::log10(arg);
                  case e_log2  : return numeric::log2 (arg);
                  case e_log1p : return numeric::log1p(arg);
                  case e_neg   : return numeric::neg  (arg);
                  case e_pos   : return numeric::pos  (arg);
                  case e_round : return numeric::round(arg);
                  case e_sin   : return numeric::sin  (arg);
                  case e_sinc  : return numeric::sinc (arg);
                  case e_sinh  : return numeric::sinh (arg);
                  case e_sqrt  : return numeric::sqrt (arg);
                  case e_tan   : return numeric::tan  (arg);
                  case e_tanh  : return numeric::tanh (arg);
                  case e_cot   : return numeric::cot  (arg);
                  case e_sec   : return numeric::sec  (arg);
                  case e_csc   : return numeric::csc  (arg);
                  case e_r2d   : return numeric::r2d  (arg);
                  case e_d2r   : return numeric::d2r  (arg);
                  case e_d2g   : return numeric::d2g  (arg);
                  case e_g2d   : return numeric::g2d  (arg);
                  case e_notl  : return numeric::notl (arg);
                  case e_sgn   : return numeric::sgn  (arg);
                  case e_erf   : return numeric::erf  (arg);
                  case e_erfc  : return numeric::erfc (arg);
                  case e_ncdf  : return numeric::ncdf (arg);
                  case e_frac  : return numeric::frac (arg);
                  case e_trunc : return numeric::trunc(arg);

                  default      : exprtk_debug(("numeric::details::process_impl<T> - Invalid unary operation.\n"));
                                 return std::numeric_limits<T>::quiet_NaN();
               }
            }

            template <typename T>
            inline T process_impl(const operator_type operation, const T arg0, const T arg1)
            {
               switch (operation)
               {
                  case e_add    : return (arg0 + arg1);
                  case e_sub    : return (arg0 - arg1);
                  case e_mul    : return (arg0 * arg1);
                  case e_div    : return (arg0 / arg1);
                  case e_mod    : return modulus<T>(arg0,arg1);
                  case e_pow    : return pow<T>(arg0,arg1);
                  case e_atan2  : return atan2<T>(arg0,arg1);
                  case e_min    : return std::min<T>(arg0,arg1);
                  case e_max    : return std::max<T>(arg0,arg1);
                  case e_logn   : return logn<T>(arg0,arg1);
                  case e_lt     : return (arg0 <  arg1) ? T(1) : T(0);
                  case e_lte    : return (arg0 <= arg1) ? T(1) : T(0);
                  case e_eq     : return std::equal_to<T>()(arg0,arg1) ? T(1) : T(0);
                  case e_ne     : return std::not_equal_to<T>()(arg0,arg1) ? T(1) : T(0);
                  case e_gte    : return (arg0 >= arg1) ? T(1) : T(0);
                  case e_gt     : return (arg0 >  arg1) ? T(1) : T(0);
                  case e_and    : return and_opr <T>(arg0,arg1);
                  case e_nand   : return nand_opr<T>(arg0,arg1);
                  case e_or     : return or_opr  <T>(arg0,arg1);
                  case e_nor    : return nor_opr <T>(arg0,arg1);
                  case e_xor    : return xor_opr <T>(arg0,arg1);
                  case e_xnor   : return xnor_opr<T>(arg0,arg1);
                  case e_root   : return root    <T>(arg0,arg1);
                  case e_roundn : return roundn  <T>(arg0,arg1);
                  case e_equal  : return equal      (arg0,arg1);
                  case e_nequal : return nequal     (arg0,arg1);
                  case e_hypot  : return hypot   <T>(arg0,arg1);
                  case e_shr    : return shr     <T>(arg0,arg1);
                  case e_shl    : return shl     <T>(arg0,arg1);

                  default       : exprtk_debug(("numeric::details::process_impl<T> - Invalid binary operation.\n"));
                                  return std::numeric_limits<T>::quiet_NaN();
               }
            }

            template <typename T>
            inline T process_impl(const operator_type operation, const T arg0, const T arg1, int_type_tag)
            {
               switch (operation)
               {
                  case e_add    : return (arg0 + arg1);
                  case e_sub    : return (arg0 - arg1);
                  case e_mul    : return (arg0 * arg1);
                  case e_div    : return (arg0 / arg1);
                  case e_mod    : return arg0 % arg1;
                  case e_pow    : return pow<T>(arg0,arg1);
                  case e_min    : return std::min<T>(arg0,arg1);
                  case e_max    : return std::max<T>(arg0,arg1);
                  case e_logn   : return logn<T>(arg0,arg1);
                  case e_lt     : return (arg0 <  arg1) ? T(1) : T(0);
                  case e_lte    : return (arg0 <= arg1) ? T(1) : T(0);
                  case e_eq     : return (arg0 == arg1) ? T(1) : T(0);
                  case e_ne     : return (arg0 != arg1) ? T(1) : T(0);
                  case e_gte    : return (arg0 >= arg1) ? T(1) : T(0);
                  case e_gt     : return (arg0 >  arg1) ? T(1) : T(0);
                  case e_and    : return ((arg0 != T(0)) && (arg1 != T(0))) ? T(1) : T(0);
                  case e_nand   : return ((arg0 != T(0)) && (arg1 != T(0))) ? T(0) : T(1);
                  case e_or     : return ((arg0 != T(0)) || (arg1 != T(0))) ? T(1) : T(0);
                  case e_nor    : return ((arg0 != T(0)) || (arg1 != T(0))) ? T(0) : T(1);
                  case e_xor    : return arg0 ^ arg1;
                  case e_xnor   : return !(arg0 ^ arg1);
                  case e_root   : return root<T>(arg0,arg1);
                  case e_equal  : return arg0 == arg1;
                  case e_nequal : return arg0 != arg1;
                  case e_hypot  : return hypot<T>(arg0,arg1);
                  case e_shr    : return arg0 >> arg1;
                  case e_shl    : return arg0 << arg1;

                  default       : exprtk_debug(("numeric::details::process_impl<IntType> - Invalid binary operation.\n"));
                                  return std::numeric_limits<T>::quiet_NaN();
               }
            }
         }

         template <typename T>
         inline T process(const operator_type operation, const T arg)
         {
            return exprtk::details::numeric::details::process_impl(operation,arg);
         }

         template <typename T>
         inline T process(const operator_type operation, const T arg0, const T arg1)
         {
            return exprtk::details::numeric::details::process_impl(operation, arg0, arg1);
         }
      }

      template <typename Node>
      struct node_collector_interface
      {
         typedef Node* node_ptr_t;
         typedef Node** node_pp_t;
         typedef std::vector<node_pp_t> noderef_list_t;

         virtual ~node_collector_interface() {}

         virtual void collect_nodes(noderef_list_t&) {}
      };

      template <typename Node>
      struct node_depth_base;

      template <typename T>
      class expression_node : public node_collector_interface<expression_node<T> >,
                              public node_depth_base<expression_node<T> >
      {
      public:

         enum node_type
         {
            e_none          , e_null          , e_constant    , e_unary        ,
            e_binary        , e_binary_ext    , e_trinary     , e_quaternary   ,
            e_vararg        , e_conditional   , e_while       , e_repeat       ,
            e_for           , e_switch        , e_mswitch     , e_return       ,
            e_retenv        , e_variable      , e_stringvar   , e_stringconst  ,
            e_stringvarrng  , e_cstringvarrng , e_strgenrange , e_strconcat    ,
            e_stringvarsize , e_strswap       , e_stringsize  , e_stringvararg ,
            e_function      , e_vafunction    , e_genfunction , e_strfunction  ,
            e_strcondition  , e_strccondition , e_add         , e_sub          ,
            e_mul           , e_div           , e_mod         , e_pow          ,
            e_lt            , e_lte           , e_gt          , e_gte          ,
            e_eq            , e_ne            , e_and         , e_nand         ,
            e_or            , e_nor           , e_xor         , e_xnor         ,
            e_in            , e_like          , e_ilike       , e_inranges     ,
            e_ipow          , e_ipowinv       , e_abs         , e_acos         ,
            e_acosh         , e_asin          , e_asinh       , e_atan         ,
            e_atanh         , e_ceil          , e_cos         , e_cosh         ,
            e_exp           , e_expm1         , e_floor       , e_log          ,
            e_log10         , e_log2          , e_log1p       , e_neg          ,
            e_pos           , e_round         , e_sin         , e_sinc         ,
            e_sinh          , e_sqrt          , e_tan         , e_tanh         ,
            e_cot           , e_sec           , e_csc         , e_r2d          ,
            e_d2r           , e_d2g           , e_g2d         , e_notl         ,
            e_sgn           , e_erf           , e_erfc        , e_ncdf         ,
            e_frac          , e_trunc         , e_uvouv       , e_vov          ,
            e_cov           , e_voc           , e_vob         , e_bov          ,
            e_cob           , e_boc           , e_vovov       , e_vovoc        ,
            e_vocov         , e_covov         , e_covoc       , e_vovovov      ,
            e_vovovoc       , e_vovocov       , e_vocovov     , e_covovov      ,
            e_covocov       , e_vocovoc       , e_covovoc     , e_vococov      ,
            e_sf3ext        , e_sf4ext        , e_nulleq      , e_strass       ,
            e_vector        , e_vecelem       , e_rbvecelem   , e_rbveccelem   ,
            e_vecdefass     , e_vecvalass     , e_vecvecass   , e_vecopvalass  ,
            e_vecopvecass   , e_vecfunc       , e_vecvecswap  , e_vecvecineq   ,
            e_vecvalineq    , e_valvecineq    , e_vecvecarith , e_vecvalarith  ,
            e_valvecarith   , e_vecunaryop    , e_vecondition , e_break        ,
            e_continue      , e_swap
         };

         typedef T value_type;
         typedef expression_node<T>* expression_ptr;
         typedef node_collector_interface<expression_node<T> > nci_t;
         typedef typename nci_t::noderef_list_t noderef_list_t;
         typedef node_depth_base<expression_node<T> > ndb_t;

         virtual ~expression_node() {}

         inline virtual T value() const
         {
            return std::numeric_limits<T>::quiet_NaN();
         }

         inline virtual expression_node<T>* branch(const std::size_t& index = 0) const
         {
            return reinterpret_cast<expression_ptr>(index * 0);
         }

         inline virtual node_type type() const
         {
            return e_none;
         }
      }; // class expression_node

      template <typename T>
      inline bool is_generally_string_node(const expression_node<T>* node);

      inline bool is_true(const double v)
      {
         return std::not_equal_to<double>()(0.0,v);
      }

      inline bool is_true(const long double v)
      {
         return std::not_equal_to<long double>()(0.0L,v);
      }

      inline bool is_true(const float v)
      {
         return std::not_equal_to<float>()(0.0f,v);
      }

      template <typename T>
      inline bool is_true(const std::complex<T>& v)
      {
         return std::not_equal_to<std::complex<T> >()(std::complex<T>(0),v);
      }

      template <typename T>
      inline bool is_true(const expression_node<T>* node)
      {
         return std::not_equal_to<T>()(T(0),node->value());
      }

      template <typename T>
      inline bool is_true(const std::pair<expression_node<T>*,bool>& node)
      {
         return std::not_equal_to<T>()(T(0),node.first->value());
      }

      template <typename T>
      inline bool is_false(const expression_node<T>* node)
      {
         return std::equal_to<T>()(T(0),node->value());
      }

      template <typename T>
      inline bool is_false(const std::pair<expression_node<T>*,bool>& node)
      {
         return std::equal_to<T>()(T(0),node.first->value());
      }

      template <typename T>
      inline bool is_unary_node(const expression_node<T>* node)
      {
         return node && (details::expression_node<T>::e_unary == node->type());
      }

      template <typename T>
      inline bool is_neg_unary_node(const expression_node<T>* node)
      {
         return node && (details::expression_node<T>::e_neg == node->type());
      }

      template <typename T>
      inline bool is_binary_node(const expression_node<T>* node)
      {
         return node && (details::expression_node<T>::e_binary == node->type());
      }

      template <typename T>
      inline bool is_variable_node(const expression_node<T>* node)
      {
         return node && (details::expression_node<T>::e_variable == node->type());
      }

      template <typename T>
      inline bool is_ivariable_node(const expression_node<T>* node)
      {
         return node &&
                (
                  details::expression_node<T>::e_variable   == node->type() ||
                  details::expression_node<T>::e_vecelem    == node->type() ||
                  details::expression_node<T>::e_rbvecelem  == node->type() ||
                  details::expression_node<T>::e_rbveccelem == node->type()
                );
      }

      template <typename T>
      inline bool is_vector_elem_node(const expression_node<T>* node)
      {
         return node && (details::expression_node<T>::e_vecelem == node->type());
      }

      template <typename T>
      inline bool is_rebasevector_elem_node(const expression_node<T>* node)
      {
         return node && (details::expression_node<T>::e_rbvecelem == node->type());
      }

      template <typename T>
      inline bool is_rebasevector_celem_node(const expression_node<T>* node)
      {
         return node && (details::expression_node<T>::e_rbveccelem == node->type());
      }

      template <typename T>
      inline bool is_vector_node(const expression_node<T>* node)
      {
         return node && (details::expression_node<T>::e_vector == node->type());
      }

      template <typename T>
      inline bool is_ivector_node(const expression_node<T>* node)
      {
         if (node)
         {
            switch (node->type())
            {
               case details::expression_node<T>::e_vector      :
               case details::expression_node<T>::e_vecvalass   :
               case details::expression_node<T>::e_vecvecass   :
               case details::expression_node<T>::e_vecopvalass :
               case details::expression_node<T>::e_vecopvecass :
               case details::expression_node<T>::e_vecvecswap  :
               case details::expression_node<T>::e_vecvecarith :
               case details::expression_node<T>::e_vecvalarith :
               case details::expression_node<T>::e_valvecarith :
               case details::expression_node<T>::e_vecunaryop  :
               case details::expression_node<T>::e_vecondition : return true;
               default                                         : return false;
            }
         }
         else
            return false;
      }

      template <typename T>
      inline bool is_constant_node(const expression_node<T>* node)
      {
         return node && (details::expression_node<T>::e_constant == node->type());
      }

      template <typename T>
      inline bool is_null_node(const expression_node<T>* node)
      {
         return node && (details::expression_node<T>::e_null == node->type());
      }

      template <typename T>
      inline bool is_break_node(const expression_node<T>* node)
      {
         return node && (details::expression_node<T>::e_break == node->type());
      }

      template <typename T>
      inline bool is_continue_node(const expression_node<T>* node)
      {
         return node && (details::expression_node<T>::e_continue == node->type());
      }

      template <typename T>
      inline bool is_swap_node(const expression_node<T>* node)
      {
         return node && (details::expression_node<T>::e_swap == node->type());
      }

      template <typename T>
      inline bool is_function(const expression_node<T>* node)
      {
         return node && (details::expression_node<T>::e_function == node->type());
      }

      template <typename T>
      inline bool is_return_node(const expression_node<T>* node)
      {
         return node && (details::expression_node<T>::e_return == node->type());
      }

      template <typename T> class unary_node;

      template <typename T>
      inline bool is_negate_node(const expression_node<T>* node)
      {
         if (node && is_unary_node(node))
         {
            return (details::e_neg == static_cast<const unary_node<T>*>(node)->operation());
         }
         else
            return false;
      }

      template <typename T>
      inline bool branch_deletable(expression_node<T>* node)
      {
         return (0 != node)             &&
                !is_variable_node(node) &&
                !is_string_node  (node) ;
      }

      template <std::size_t N, typename T>
      inline bool all_nodes_valid(expression_node<T>* (&b)[N])
      {
         for (std::size_t i = 0; i < N; ++i)
         {
            if (0 == b[i]) return false;
         }

         return true;
      }

      template <typename T,
                typename Allocator,
                template <typename, typename> class Sequence>
      inline bool all_nodes_valid(const Sequence<expression_node<T>*,Allocator>& b)
      {
         for (std::size_t i = 0; i < b.size(); ++i)
         {
            if (0 == b[i]) return false;
         }

         return true;
      }

      template <std::size_t N, typename T>
      inline bool all_nodes_variables(expression_node<T>* (&b)[N])
      {
         for (std::size_t i = 0; i < N; ++i)
         {
            if (0 == b[i])
               return false;
            else if (!is_variable_node(b[i]))
               return false;
         }

         return true;
      }

      template <typename T,
                typename Allocator,
                template <typename, typename> class Sequence>
      inline bool all_nodes_variables(Sequence<expression_node<T>*,Allocator>& b)
      {
         for (std::size_t i = 0; i < b.size(); ++i)
         {
            if (0 == b[i])
               return false;
            else if (!is_variable_node(b[i]))
               return false;
         }

         return true;
      }

      template <typename Node>
      class node_collection_destructor
      {
      public:

         typedef node_collector_interface<Node> nci_t;

         typedef typename nci_t::node_ptr_t     node_ptr_t;
         typedef typename nci_t::node_pp_t      node_pp_t;
         typedef typename nci_t::noderef_list_t noderef_list_t;

         static void delete_nodes(node_ptr_t& root)
         {
            std::vector<node_pp_t> node_delete_list;
            node_delete_list.reserve(1000);

            collect_nodes(root, node_delete_list);

            for (std::size_t i = 0; i < node_delete_list.size(); ++i)
            {
               node_ptr_t& node = *node_delete_list[i];
               exprtk_debug(("ncd::delete_nodes() - deleting: %p\n", static_cast<void*>(node)));
               delete node;
               node = reinterpret_cast<node_ptr_t>(0);
            }
         }

      private:

         static void collect_nodes(node_ptr_t& root, noderef_list_t& node_delete_list)
         {
            std::deque<node_ptr_t> node_list;
            node_list.push_back(root);
            node_delete_list.push_back(&root);

            noderef_list_t child_node_delete_list;
            child_node_delete_list.reserve(1000);

            while (!node_list.empty())
            {
               node_list.front()->collect_nodes(child_node_delete_list);

               if (!child_node_delete_list.empty())
               {
                  for (std::size_t i = 0; i < child_node_delete_list.size(); ++i)
                  {
                     node_pp_t& node = child_node_delete_list[i];

                     if (0 == (*node))
                     {
                        exprtk_debug(("ncd::collect_nodes() - null node encountered.\n"));
                     }

                     node_list.push_back(*node);
                  }

                  node_delete_list.insert(
                     node_delete_list.end(),
                     child_node_delete_list.begin(), child_node_delete_list.end());

                  child_node_delete_list.clear();
               }

               node_list.pop_front();
            }

            std::reverse(node_delete_list.begin(), node_delete_list.end());
         }
      };

      template <typename NodeAllocator, typename T, std::size_t N>
      inline void free_all_nodes(NodeAllocator& node_allocator, expression_node<T>* (&b)[N])
      {
         for (std::size_t i = 0; i < N; ++i)
         {
            free_node(node_allocator,b[i]);
         }
      }

      template <typename NodeAllocator,
                typename T,
                typename Allocator,
                template <typename, typename> class Sequence>
      inline void free_all_nodes(NodeAllocator& node_allocator, Sequence<expression_node<T>*,Allocator>& b)
      {
         for (std::size_t i = 0; i < b.size(); ++i)
         {
            free_node(node_allocator,b[i]);
         }

         b.clear();
      }

      template <typename NodeAllocator, typename T>
      inline void free_node(NodeAllocator&, expression_node<T>*& node)
      {
         if ((0 == node) || is_variable_node(node) || is_string_node(node))
         {
            return;
         }

         node_collection_destructor<expression_node<T> >
            ::delete_nodes(node);
      }

      template <typename T>
      inline void destroy_node(expression_node<T>*& node)
      {
         if (0 != node)
         {
            node_collection_destructor<expression_node<T> >
               ::delete_nodes(node);
         }
      }

      template <typename Node>
      struct node_depth_base
      {
         typedef Node* node_ptr_t;
         typedef std::pair<node_ptr_t,bool> nb_pair_t;

         node_depth_base()
         : depth_set(false)
         , depth(0)
         {}

         virtual ~node_depth_base() {}

         virtual std::size_t node_depth() const { return 1; }

         std::size_t compute_node_depth(const Node* const& node) const
         {
            if (!depth_set)
            {
               depth = 1 + (node ? node->node_depth() : 0);
               depth_set = true;
            }

            return depth;
         }

         std::size_t compute_node_depth(const nb_pair_t& branch) const
         {
            if (!depth_set)
            {
               depth = 1 + (branch.first ? branch.first->node_depth() : 0);
               depth_set = true;
            }

            return depth;
         }

         template <std::size_t N>
         std::size_t compute_node_depth(const nb_pair_t (&branch)[N]) const
         {
            if (!depth_set)
            {
               depth = 0;
               for (std::size_t i = 0; i < N; ++i)
               {
                  if (branch[i].first)
                  {
                     depth = std::max(depth,branch[i].first->node_depth());
                  }
               }
               depth += 1;
               depth_set = true;
            }

            return depth;
         }

         template <typename BranchType>
         std::size_t compute_node_depth(const BranchType& n0, const BranchType& n1) const
         {
            if (!depth_set)
            {
               depth = 1 + std::max(compute_node_depth(n0), compute_node_depth(n1));
               depth_set = true;
            }

            return depth;
         }

         template <typename BranchType>
         std::size_t compute_node_depth(const BranchType& n0, const BranchType& n1,
                                        const BranchType& n2) const
         {
            if (!depth_set)
            {
               depth = 1 + std::max(
                              std::max(compute_node_depth(n0), compute_node_depth(n1)),
                              compute_node_depth(n2));
               depth_set = true;
            }

            return depth;
         }

         template <typename BranchType>
         std::size_t compute_node_depth(const BranchType& n0, const BranchType& n1,
                                        const BranchType& n2, const BranchType& n3) const
         {
            if (!depth_set)
            {
               depth = 1 + std::max(
                           std::max(compute_node_depth(n0), compute_node_depth(n1)),
                           std::max(compute_node_depth(n2), compute_node_depth(n3)));
               depth_set = true;
            }

            return depth;
         }

         template <typename Allocator,
                   template <typename, typename> class Sequence>
         std::size_t compute_node_depth(const Sequence<node_ptr_t, Allocator>& branch_list) const
         {
            if (!depth_set)
            {
               for (std::size_t i = 0; i < branch_list.size(); ++i)
               {
                  if (branch_list[i])
                  {
                     depth = std::max(depth, compute_node_depth(branch_list[i]));
                  }
               }
               depth_set = true;
            }

            return depth;
         }

         template <typename Allocator,
                   template <typename, typename> class Sequence>
         std::size_t compute_node_depth(const Sequence<nb_pair_t,Allocator>& branch_list) const
         {
            if (!depth_set)
            {
               for (std::size_t i = 0; i < branch_list.size(); ++i)
               {
                  if (branch_list[i].first)
                  {
                     depth = std::max(depth, compute_node_depth(branch_list[i].first));
                  }
               }
               depth_set = true;
            }

            return depth;
         }

         mutable bool depth_set;
         mutable std::size_t depth;

         template <typename NodeSequence>
         void collect(node_ptr_t const& node,
                      const bool deletable,
                      NodeSequence& delete_node_list) const
         {
            if ((0 != node) && deletable)
            {
               delete_node_list.push_back(const_cast<node_ptr_t*>(&node));
            }
         }

         template <typename NodeSequence>
         void collect(const nb_pair_t& branch,
                      NodeSequence& delete_node_list) const
         {
            collect(branch.first, branch.second, delete_node_list);
         }

         template <typename NodeSequence>
         void collect(Node*& node,
                      NodeSequence& delete_node_list) const
         {
            collect(node, branch_deletable(node), delete_node_list);
         }

         template <std::size_t N, typename NodeSequence>
         void collect(const nb_pair_t(&branch)[N],
                      NodeSequence& delete_node_list) const
         {
            for (std::size_t i = 0; i < N; ++i)
            {
               collect(branch[i].first, branch[i].second, delete_node_list);
            }
         }

         template <typename Allocator,
                   template <typename, typename> class Sequence,
                   typename NodeSequence>
         void collect(const Sequence<nb_pair_t, Allocator>& branch,
                      NodeSequence& delete_node_list) const
         {
            for (std::size_t i = 0; i < branch.size(); ++i)
            {
               collect(branch[i].first, branch[i].second, delete_node_list);
            }
         }

         template <typename Allocator,
                   template <typename, typename> class Sequence,
                   typename NodeSequence>
         void collect(const Sequence<node_ptr_t, Allocator>& branch_list,
                      NodeSequence& delete_node_list) const
         {
            for (std::size_t i = 0; i < branch_list.size(); ++i)
            {
               collect(branch_list[i], branch_deletable(branch_list[i]), delete_node_list);
            }
         }

         template <typename Boolean,
                   typename AllocatorT,
                   typename AllocatorB,
                   template <typename, typename> class Sequence,
                   typename NodeSequence>
         void collect(const Sequence<node_ptr_t, AllocatorT>& branch_list,
                      const Sequence<Boolean, AllocatorB>& branch_deletable_list,
                      NodeSequence& delete_node_list) const
         {
            for (std::size_t i = 0; i < branch_list.size(); ++i)
            {
               collect(branch_list[i], branch_deletable_list[i], delete_node_list);
            }
         }
      };

      template <typename Type>
      class vector_holder
      {
      private:

         typedef Type value_type;
         typedef value_type* value_ptr;
         typedef const value_ptr const_value_ptr;

         class vector_holder_base
         {
         public:

            virtual ~vector_holder_base() {}

            inline value_ptr operator[](const std::size_t& index) const
            {
               return value_at(index);
            }

            inline std::size_t size() const
            {
               return vector_size();
            }

            inline value_ptr data() const
            {
               return value_at(0);
            }

            virtual inline bool rebaseable() const
            {
               return false;
            }

            virtual void set_ref(value_ptr*) {}

         protected:

            virtual value_ptr value_at(const std::size_t&) const = 0;
            virtual std::size_t vector_size()              const = 0;
         };

         class array_vector_impl : public vector_holder_base
         {
         public:

            array_vector_impl(const Type* vec, const std::size_t& vec_size)
            : vec_(vec)
            , size_(vec_size)
            {}

         protected:

            value_ptr value_at(const std::size_t& index) const
            {
               if (index < size_)
                  return const_cast<const_value_ptr>(vec_ + index);
               else
                  return const_value_ptr(0);
            }

            std::size_t vector_size() const
            {
               return size_;
            }

         private:

            array_vector_impl(const array_vector_impl&) exprtk_delete;
            array_vector_impl& operator=(const array_vector_impl&) exprtk_delete;

            const Type* vec_;
            const std::size_t size_;
         };

         template <typename Allocator,
                   template <typename, typename> class Sequence>
         class sequence_vector_impl : public vector_holder_base
         {
         public:

            typedef Sequence<Type,Allocator> sequence_t;

            sequence_vector_impl(sequence_t& seq)
            : sequence_(seq)
            {}

         protected:

            value_ptr value_at(const std::size_t& index) const
            {
               return (index < sequence_.size()) ? (&sequence_[index]) : const_value_ptr(0);
            }

            std::size_t vector_size() const
            {
               return sequence_.size();
            }

         private:

            sequence_vector_impl(const sequence_vector_impl&) exprtk_delete;
            sequence_vector_impl& operator=(const sequence_vector_impl&) exprtk_delete;

            sequence_t& sequence_;
         };

         class vector_view_impl : public vector_holder_base
         {
         public:

            typedef exprtk::vector_view<Type> vector_view_t;

            vector_view_impl(vector_view_t& vec_view)
            : vec_view_(vec_view)
            {}

            void set_ref(value_ptr* ref)
            {
               vec_view_.set_ref(ref);
            }

            virtual inline bool rebaseable() const
            {
               return true;
            }

         protected:

            value_ptr value_at(const std::size_t& index) const
            {
               return (index < vec_view_.size()) ? (&vec_view_[index]) : const_value_ptr(0);
            }

            std::size_t vector_size() const
            {
               return vec_view_.size();
            }

         private:

            vector_view_impl(const vector_view_impl&) exprtk_delete;
            vector_view_impl& operator=(const vector_view_impl&) exprtk_delete;

            vector_view_t& vec_view_;
         };

      public:

         typedef typename details::vec_data_store<Type> vds_t;

         vector_holder(Type* vec, const std::size_t& vec_size)
         : vector_holder_base_(new(buffer)array_vector_impl(vec,vec_size))
         {}

         vector_holder(const vds_t& vds)
         : vector_holder_base_(new(buffer)array_vector_impl(vds.data(),vds.size()))
         {}

         template <typename Allocator>
         vector_holder(std::vector<Type,Allocator>& vec)
         : vector_holder_base_(new(buffer)sequence_vector_impl<Allocator,std::vector>(vec))
         {}

         vector_holder(exprtk::vector_view<Type>& vec)
         : vector_holder_base_(new(buffer)vector_view_impl(vec))
         {}

         inline value_ptr operator[](const std::size_t& index) const
         {
            return (*vector_holder_base_)[index];
         }

         inline std::size_t size() const
         {
            return vector_holder_base_->size();
         }

         inline value_ptr data() const
         {
            return vector_holder_base_->data();
         }

         void set_ref(value_ptr* ref)
         {
            vector_holder_base_->set_ref(ref);
         }

         bool rebaseable() const
         {
            return vector_holder_base_->rebaseable();
         }

      private:

         mutable vector_holder_base* vector_holder_base_;
         uchar_t buffer[64];
      };

      template <typename T>
      class null_node exprtk_final : public expression_node<T>
      {
      public:

         inline T value() const exprtk_override
         {
            return std::numeric_limits<T>::quiet_NaN();
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_null;
         }
      };

      template <typename T, std::size_t N>
      inline void construct_branch_pair(std::pair<expression_node<T>*,bool> (&branch)[N],
                                        expression_node<T>* b,
                                        const std::size_t& index)
      {
         if (b && (index < N))
         {
            branch[index] = std::make_pair(b,branch_deletable(b));
         }
      }

      template <typename T>
      inline void construct_branch_pair(std::pair<expression_node<T>*,bool>& branch, expression_node<T>* b)
      {
         if (b)
         {
            branch = std::make_pair(b,branch_deletable(b));
         }
      }

      template <std::size_t N, typename T>
      inline void init_branches(std::pair<expression_node<T>*,bool> (&branch)[N],
                                expression_node<T>* b0,
                                expression_node<T>* b1 = reinterpret_cast<expression_node<T>*>(0),
                                expression_node<T>* b2 = reinterpret_cast<expression_node<T>*>(0),
                                expression_node<T>* b3 = reinterpret_cast<expression_node<T>*>(0),
                                expression_node<T>* b4 = reinterpret_cast<expression_node<T>*>(0),
                                expression_node<T>* b5 = reinterpret_cast<expression_node<T>*>(0),
                                expression_node<T>* b6 = reinterpret_cast<expression_node<T>*>(0),
                                expression_node<T>* b7 = reinterpret_cast<expression_node<T>*>(0),
                                expression_node<T>* b8 = reinterpret_cast<expression_node<T>*>(0),
                                expression_node<T>* b9 = reinterpret_cast<expression_node<T>*>(0))
      {
         construct_branch_pair(branch, b0, 0);
         construct_branch_pair(branch, b1, 1);
         construct_branch_pair(branch, b2, 2);
         construct_branch_pair(branch, b3, 3);
         construct_branch_pair(branch, b4, 4);
         construct_branch_pair(branch, b5, 5);
         construct_branch_pair(branch, b6, 6);
         construct_branch_pair(branch, b7, 7);
         construct_branch_pair(branch, b8, 8);
         construct_branch_pair(branch, b9, 9);
      }

      template <typename T>
      class null_eq_node exprtk_final : public expression_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef std::pair<expression_ptr,bool> branch_t;

         explicit null_eq_node(expression_ptr branch, const bool equality = true)
         : equality_(equality)
         {
            construct_branch_pair(branch_, branch);
         }

         inline T value() const exprtk_override
         {
            assert(branch_.first);

            const T v = branch_.first->value();
            const bool result = details::numeric::is_nan(v);

            if (result)
               return (equality_) ? T(1) : T(0);
            else
               return (equality_) ? T(0) : T(1);
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_nulleq;
         }

         inline expression_node<T>* branch(const std::size_t&) const exprtk_override
         {
            return branch_.first;
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(branch_, node_delete_list);
         }

         std::size_t node_depth() const exprtk_override
         {
            return expression_node<T>::ndb_t::compute_node_depth(branch_);
         }

      private:

         bool equality_;
         branch_t branch_;
      };

      template <typename T>
      class literal_node exprtk_final : public expression_node<T>
      {
      public:

         explicit literal_node(const T& v)
         : value_(v)
         {}

         inline T value() const exprtk_override
         {
            return value_;
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_constant;
         }

         inline expression_node<T>* branch(const std::size_t&) const exprtk_override
         {
            return reinterpret_cast<expression_node<T>*>(0);
         }

      private:

         literal_node(const literal_node<T>&) exprtk_delete;
         literal_node<T>& operator=(const literal_node<T>&) exprtk_delete;

         const T value_;
      };

      template <typename T>
      struct range_pack;

      template <typename T>
      struct range_data_type;

      template <typename T>
      class range_interface
      {
      public:

         typedef range_pack<T> range_t;

         virtual ~range_interface() {}

         virtual range_t& range_ref() = 0;

         virtual const range_t& range_ref() const = 0;
      };

      template <typename T>
      class unary_node : public expression_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef std::pair<expression_ptr,bool> branch_t;

         unary_node(const operator_type& opr, expression_ptr branch)
         : operation_(opr)
         {
            construct_branch_pair(branch_,branch);
         }

         inline T value() const exprtk_override
         {
            assert(branch_.first);

            const T arg = branch_.first->value();

            return numeric::process<T>(operation_,arg);
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_unary;
         }

         inline operator_type operation()
         {
            return operation_;
         }

         inline expression_node<T>* branch(const std::size_t&) const exprtk_override
         {
            return branch_.first;
         }

         inline void release()
         {
            branch_.second = false;
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(branch_, node_delete_list);
         }

         std::size_t node_depth() const exprtk_override exprtk_final
         {
            return expression_node<T>::ndb_t::compute_node_depth(branch_);
         }

      protected:

         operator_type operation_;
         branch_t branch_;
      };

      template <typename T>
      class binary_node : public expression_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef std::pair<expression_ptr,bool> branch_t;

         binary_node(const operator_type& opr,
                     expression_ptr branch0,
                     expression_ptr branch1)
         : operation_(opr)
         {
            init_branches<2>(branch_, branch0, branch1);
         }

         inline T value() const exprtk_override
         {
            assert(branch_[0].first);
            assert(branch_[1].first);

            const T arg0 = branch_[0].first->value();
            const T arg1 = branch_[1].first->value();

            return numeric::process<T>(operation_,arg0,arg1);
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_binary;
         }

         inline operator_type operation()
         {
            return operation_;
         }

         inline expression_node<T>* branch(const std::size_t& index = 0) const exprtk_override
         {
            if (0 == index)
               return branch_[0].first;
            else if (1 == index)
               return branch_[1].first;
            else
               return reinterpret_cast<expression_ptr>(0);
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(branch_, node_delete_list);
         }

         std::size_t node_depth() const exprtk_override exprtk_final
         {
            return expression_node<T>::ndb_t::template compute_node_depth<2>(branch_);
         }

      protected:

         operator_type operation_;
         branch_t branch_[2];
      };

      template <typename T, typename Operation>
      class binary_ext_node exprtk_final : public expression_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef std::pair<expression_ptr,bool> branch_t;

         binary_ext_node(expression_ptr branch0, expression_ptr branch1)
         {
            init_branches<2>(branch_, branch0, branch1);
         }

         inline T value() const exprtk_override
         {
            assert(branch_[0].first);
            assert(branch_[1].first);

            const T arg0 = branch_[0].first->value();
            const T arg1 = branch_[1].first->value();

            return Operation::process(arg0,arg1);
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_binary_ext;
         }

         inline operator_type operation()
         {
            return Operation::operation();
         }

         inline expression_node<T>* branch(const std::size_t& index = 0) const exprtk_override
         {
            if (0 == index)
               return branch_[0].first;
            else if (1 == index)
               return branch_[1].first;
            else
               return reinterpret_cast<expression_ptr>(0);
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(branch_, node_delete_list);
         }

         std::size_t node_depth() const exprtk_override
         {
            return expression_node<T>::ndb_t::template compute_node_depth<2>(branch_);
         }

      protected:

         branch_t branch_[2];
      };

      template <typename T>
      class trinary_node : public expression_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef std::pair<expression_ptr,bool> branch_t;

         trinary_node(const operator_type& opr,
                      expression_ptr branch0,
                      expression_ptr branch1,
                      expression_ptr branch2)
         : operation_(opr)
         {
            init_branches<3>(branch_, branch0, branch1, branch2);
         }

         inline T value() const exprtk_override
         {
            assert(branch_[0].first);
            assert(branch_[1].first);
            assert(branch_[2].first);

            const T arg0 = branch_[0].first->value();
            const T arg1 = branch_[1].first->value();
            const T arg2 = branch_[2].first->value();

            switch (operation_)
            {
               case e_inrange : return (arg1 < arg0) ? T(0) : ((arg1 > arg2) ? T(0) : T(1));

               case e_clamp   : return (arg1 < arg0) ? arg0 : (arg1 > arg2 ? arg2 : arg1);

               case e_iclamp  : if ((arg1 <= arg0) || (arg1 >= arg2))
                                   return arg1;
                                else
                                   return ((T(2) * arg1  <= (arg2 + arg0)) ? arg0 : arg2);

               default        : exprtk_debug(("trinary_node::value() - Error: Invalid operation\n"));
                                return std::numeric_limits<T>::quiet_NaN();
            }
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_trinary;
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(branch_, node_delete_list);
         }

         std::size_t node_depth() const exprtk_override exprtk_final
         {
            return expression_node<T>::ndb_t::template compute_node_depth<3>(branch_);
         }

      protected:

         operator_type operation_;
         branch_t branch_[3];
      };

      template <typename T>
      class quaternary_node : public expression_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef std::pair<expression_ptr,bool> branch_t;

         quaternary_node(const operator_type& opr,
                         expression_ptr branch0,
                         expression_ptr branch1,
                         expression_ptr branch2,
                         expression_ptr branch3)
         : operation_(opr)
         {
            init_branches<4>(branch_, branch0, branch1, branch2, branch3);
         }

         inline T value() const exprtk_override
         {
            return std::numeric_limits<T>::quiet_NaN();
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_quaternary;
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(branch_, node_delete_list);
         }

         std::size_t node_depth() const exprtk_override exprtk_final
         {
            return expression_node<T>::ndb_t::template compute_node_depth<4>(branch_);
         }

      protected:

         operator_type operation_;
         branch_t branch_[4];
      };

      template <typename T>
      class conditional_node exprtk_final : public expression_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef std::pair<expression_ptr,bool> branch_t;

         conditional_node(expression_ptr condition,
                          expression_ptr consequent,
                          expression_ptr alternative)
         {
            construct_branch_pair(condition_  , condition  );
            construct_branch_pair(consequent_ , consequent );
            construct_branch_pair(alternative_, alternative);
         }

         inline T value() const exprtk_override
         {
            assert(condition_  .first);
            assert(consequent_ .first);
            assert(alternative_.first);

            if (is_true(condition_))
               return consequent_.first->value();
            else
               return alternative_.first->value();
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_conditional;
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(condition_   , node_delete_list);
            expression_node<T>::ndb_t::collect(consequent_  , node_delete_list);
            expression_node<T>::ndb_t::collect(alternative_ , node_delete_list);
         }

         std::size_t node_depth() const exprtk_override
         {
            return expression_node<T>::ndb_t::compute_node_depth
               (condition_, consequent_, alternative_);
         }

      private:

         branch_t condition_;
         branch_t consequent_;
         branch_t alternative_;
      };

      template <typename T>
      class cons_conditional_node exprtk_final : public expression_node<T>
      {
      public:

         // Consequent only conditional statement node
         typedef expression_node<T>* expression_ptr;
         typedef std::pair<expression_ptr,bool> branch_t;

         cons_conditional_node(expression_ptr condition,
                               expression_ptr consequent)
         {
            construct_branch_pair(condition_ , condition );
            construct_branch_pair(consequent_, consequent);
         }

         inline T value() const exprtk_override
         {
            assert(condition_ .first);
            assert(consequent_.first);

            if (is_true(condition_))
               return consequent_.first->value();
            else
               return std::numeric_limits<T>::quiet_NaN();
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_conditional;
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(condition_  , node_delete_list);
            expression_node<T>::ndb_t::collect(consequent_ , node_delete_list);
         }

         std::size_t node_depth() const exprtk_override
         {
            return expression_node<T>::ndb_t::
               compute_node_depth(condition_, consequent_);
         }

      private:

         branch_t condition_;
         branch_t consequent_;
      };

      struct loop_runtime_checker
      {
         loop_runtime_checker(loop_runtime_check_ptr loop_runtime_check,
                              loop_runtime_check::loop_types lp_typ = loop_runtime_check::e_invalid)
         : iteration_count_(0)
         , loop_runtime_check_(loop_runtime_check)
         , max_loop_iterations_(loop_runtime_check_->max_loop_iterations)
         , loop_type_(lp_typ)
         {
            assert(loop_runtime_check_);
         }

         inline void reset(const _uint64_t initial_value = 0) const
         {
            iteration_count_ = initial_value;
         }

         inline bool check() const
         {
            if (
                 (0 == loop_runtime_check_) ||
                 (++iteration_count_ <= max_loop_iterations_)
               )
            {
               return true;
            }

            loop_runtime_check::violation_context ctxt;
            ctxt.loop      = loop_type_;
            ctxt.violation = loop_runtime_check::e_iteration_count;

            loop_runtime_check_->handle_runtime_violation(ctxt);

            return false;
         }

         mutable _uint64_t iteration_count_;
         mutable loop_runtime_check_ptr loop_runtime_check_;
         const details::_uint64_t& max_loop_iterations_;
         loop_runtime_check::loop_types loop_type_;
      };

      template <typename T>
      class while_loop_node : public expression_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef std::pair<expression_ptr,bool> branch_t;

         while_loop_node(expression_ptr condition,
                         expression_ptr loop_body)
         {
            construct_branch_pair(condition_, condition);
            construct_branch_pair(loop_body_, loop_body);
         }

         inline T value() const exprtk_override
         {
            assert(condition_.first);
            assert(loop_body_.first);

            T result = T(0);

            while (is_true(condition_))
            {
               result = loop_body_.first->value();
            }

            return result;
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_while;
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(condition_ , node_delete_list);
            expression_node<T>::ndb_t::collect(loop_body_ , node_delete_list);
         }

         std::size_t node_depth() const exprtk_override
         {
            return expression_node<T>::ndb_t::compute_node_depth(condition_, loop_body_);
         }

      protected:

         branch_t condition_;
         branch_t loop_body_;
      };

      template <typename T>
      class while_loop_rtc_node exprtk_final
                                : public while_loop_node<T>
                                , public loop_runtime_checker
      {
      public:

         typedef while_loop_node<T>  parent_t;
         typedef expression_node<T>* expression_ptr;

         while_loop_rtc_node(expression_ptr condition,
                             expression_ptr loop_body,
                             loop_runtime_check_ptr loop_rt_chk)
         : parent_t(condition, loop_body)
         , loop_runtime_checker(loop_rt_chk, loop_runtime_check::e_while_loop)
         {}

         inline T value() const exprtk_override
         {
            assert(parent_t::condition_.first);
            assert(parent_t::loop_body_.first);

            T result = T(0);

            loop_runtime_checker::reset();

            while (is_true(parent_t::condition_) && loop_runtime_checker::check())
            {
               result = parent_t::loop_body_.first->value();
            }

            return result;
         }
      };

      template <typename T>
      class repeat_until_loop_node : public expression_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef std::pair<expression_ptr,bool> branch_t;

         repeat_until_loop_node(expression_ptr condition,
                                expression_ptr loop_body)
         {
            construct_branch_pair(condition_, condition);
            construct_branch_pair(loop_body_, loop_body);
         }

         inline T value() const exprtk_override
         {
            assert(condition_.first);
            assert(loop_body_.first);

            T result = T(0);

            do
            {
               result = loop_body_.first->value();
            }
            while (is_false(condition_.first));

            return result;
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_repeat;
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(condition_ , node_delete_list);
            expression_node<T>::ndb_t::collect(loop_body_ , node_delete_list);
         }

         std::size_t node_depth() const exprtk_override
         {
            return expression_node<T>::ndb_t::compute_node_depth(condition_, loop_body_);
         }

      protected:

         branch_t condition_;
         branch_t loop_body_;
      };

      template <typename T>
      class repeat_until_loop_rtc_node exprtk_final
                                       : public repeat_until_loop_node<T>
                                       , public loop_runtime_checker
      {
      public:

         typedef repeat_until_loop_node<T> parent_t;
         typedef expression_node<T>*       expression_ptr;

         repeat_until_loop_rtc_node(expression_ptr condition,
                                    expression_ptr loop_body,
                                    loop_runtime_check_ptr loop_rt_chk)
         : parent_t(condition, loop_body)
         , loop_runtime_checker(loop_rt_chk, loop_runtime_check::e_repeat_until_loop)
         {}

         inline T value() const exprtk_override
         {
            assert(parent_t::condition_.first);
            assert(parent_t::loop_body_.first);

            T result = T(0);

            loop_runtime_checker::reset(1);

            do
            {
               result = parent_t::loop_body_.first->value();
            }
            while (is_false(parent_t::condition_.first) && loop_runtime_checker::check());

            return result;
         }
      };

      template <typename T>
      class for_loop_node : public expression_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef std::pair<expression_ptr,bool> branch_t;

         for_loop_node(expression_ptr initialiser,
                       expression_ptr condition,
                       expression_ptr incrementor,
                       expression_ptr loop_body)
         {
            construct_branch_pair(initialiser_, initialiser);
            construct_branch_pair(condition_  , condition  );
            construct_branch_pair(incrementor_, incrementor);
            construct_branch_pair(loop_body_  , loop_body  );
         }

         inline T value() const exprtk_override
         {
            assert(condition_.first);
            assert(loop_body_.first);

            T result = T(0);

            if (initialiser_.first)
               initialiser_.first->value();

            if (incrementor_.first)
            {
               while (is_true(condition_))
               {
                  result = loop_body_.first->value();
                  incrementor_.first->value();
               }
            }
            else
            {
               while (is_true(condition_))
               {
                  result = loop_body_.first->value();
               }
            }

            return result;
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_for;
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(initialiser_ , node_delete_list);
            expression_node<T>::ndb_t::collect(condition_   , node_delete_list);
            expression_node<T>::ndb_t::collect(incrementor_ , node_delete_list);
            expression_node<T>::ndb_t::collect(loop_body_   , node_delete_list);
         }

         std::size_t node_depth() const exprtk_override
         {
            return expression_node<T>::ndb_t::compute_node_depth
               (initialiser_, condition_, incrementor_, loop_body_);
         }

      protected:

         branch_t initialiser_;
         branch_t condition_  ;
         branch_t incrementor_;
         branch_t loop_body_  ;
      };

      template <typename T>
      class for_loop_rtc_node exprtk_final
                              : public for_loop_node<T>
                              , public loop_runtime_checker
      {
      public:

         typedef for_loop_node<T>    parent_t;
         typedef expression_node<T>* expression_ptr;

         for_loop_rtc_node(expression_ptr initialiser,
                           expression_ptr condition,
                           expression_ptr incrementor,
                           expression_ptr loop_body,
                           loop_runtime_check_ptr loop_rt_chk)
         : parent_t(initialiser, condition, incrementor, loop_body)
         , loop_runtime_checker(loop_rt_chk, loop_runtime_check::e_for_loop)
         {}

         inline T value() const exprtk_override
         {
            assert(parent_t::condition_.first);
            assert(parent_t::loop_body_.first);

            T result = T(0);

            loop_runtime_checker::reset();

            if (parent_t::initialiser_.first)
               parent_t::initialiser_.first->value();

            if (parent_t::incrementor_.first)
            {
               while (is_true(parent_t::condition_) && loop_runtime_checker::check())
               {
                  result = parent_t::loop_body_.first->value();
                  parent_t::incrementor_.first->value();
               }
            }
            else
            {
               while (is_true(parent_t::condition_) && loop_runtime_checker::check())
               {
                  result = parent_t::loop_body_.first->value();
               }
            }

            return result;
         }
      };

      template <typename T>
      class switch_node : public expression_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef std::pair<expression_ptr,bool> branch_t;

         template <typename Allocator,
                   template <typename, typename> class Sequence>
         explicit switch_node(const Sequence<expression_ptr,Allocator>& arg_list)
         {
            if (1 != (arg_list.size() & 1))
               return;

            arg_list_.resize(arg_list.size());

            for (std::size_t i = 0; i < arg_list.size(); ++i)
            {
               if (arg_list[i])
               {
                  construct_branch_pair(arg_list_[i], arg_list[i]);
               }
               else
               {
                  arg_list_.clear();
                  return;
               }
            }
         }

         inline T value() const exprtk_override
         {
            if (!arg_list_.empty())
            {
               const std::size_t upper_bound = (arg_list_.size() - 1);

               for (std::size_t i = 0; i < upper_bound; i += 2)
               {
                  expression_ptr condition  = arg_list_[i    ].first;
                  expression_ptr consequent = arg_list_[i + 1].first;

                  if (is_true(condition))
                  {
                     return consequent->value();
                  }
               }

               return arg_list_[upper_bound].first->value();
            }
            else
               return std::numeric_limits<T>::quiet_NaN();
         }

         inline typename expression_node<T>::node_type type() const exprtk_override exprtk_final
         {
            return expression_node<T>::e_switch;
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(arg_list_, node_delete_list);
         }

         std::size_t node_depth() const exprtk_override exprtk_final
         {
            return expression_node<T>::ndb_t::compute_node_depth(arg_list_);
         }

      protected:

         std::vector<branch_t> arg_list_;
      };

      template <typename T, typename Switch_N>
      class switch_n_node exprtk_final : public switch_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;

         template <typename Allocator,
                   template <typename, typename> class Sequence>
         explicit switch_n_node(const Sequence<expression_ptr,Allocator>& arg_list)
         : switch_node<T>(arg_list)
         {}

         inline T value() const exprtk_override
         {
            return Switch_N::process(switch_node<T>::arg_list_);
         }
      };

      template <typename T>
      class multi_switch_node exprtk_final : public expression_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef std::pair<expression_ptr,bool> branch_t;

         template <typename Allocator,
                   template <typename, typename> class Sequence>
         explicit multi_switch_node(const Sequence<expression_ptr,Allocator>& arg_list)
         {
            if (0 != (arg_list.size() & 1))
               return;

            arg_list_.resize(arg_list.size());

            for (std::size_t i = 0; i < arg_list.size(); ++i)
            {
               if (arg_list[i])
               {
                  construct_branch_pair(arg_list_[i], arg_list[i]);
               }
               else
               {
                  arg_list_.clear();
                  return;
               }
            }
         }

         inline T value() const exprtk_override
         {
            T result = T(0);

            if (arg_list_.empty())
            {
               return std::numeric_limits<T>::quiet_NaN();
            }

            const std::size_t upper_bound = (arg_list_.size() - 1);

            for (std::size_t i = 0; i < upper_bound; i += 2)
            {
               expression_ptr condition  = arg_list_[i    ].first;
               expression_ptr consequent = arg_list_[i + 1].first;

               if (is_true(condition))
               {
                  result = consequent->value();
               }
            }

            return result;
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_mswitch;
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(arg_list_, node_delete_list);
         }

         std::size_t node_depth() const exprtk_override exprtk_final
         {
            return expression_node<T>::ndb_t::compute_node_depth(arg_list_);
         }

      private:

         std::vector<branch_t> arg_list_;
      };

      template <typename T>
      class ivariable
      {
      public:

         virtual ~ivariable() {}

         virtual T& ref() = 0;
         virtual const T& ref() const = 0;
      };

      template <typename T>
      class variable_node exprtk_final
                          : public expression_node<T>,
                            public ivariable      <T>
      {
      public:

         static T null_value;

         explicit variable_node()
         : value_(&null_value)
         {}

         explicit variable_node(T& v)
         : value_(&v)
         {}

         inline bool operator <(const variable_node<T>& v) const
         {
            return this < (&v);
         }

         inline T value() const exprtk_override
         {
            return (*value_);
         }

         inline T& ref() exprtk_override
         {
            return (*value_);
         }

         inline const T& ref() const exprtk_override
         {
            return (*value_);
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_variable;
         }

      private:

         T* value_;
      };

      template <typename T>
      T variable_node<T>::null_value = T(std::numeric_limits<T>::quiet_NaN());

      template <typename T>
      struct range_pack
      {
         typedef expression_node<T>*           expression_node_ptr;
         typedef std::pair<std::size_t,std::size_t> cached_range_t;

         range_pack()
         : n0_e (std::make_pair(false,expression_node_ptr(0)))
         , n1_e (std::make_pair(false,expression_node_ptr(0)))
         , n0_c (std::make_pair(false,0))
         , n1_c (std::make_pair(false,0))
         , cache(std::make_pair(0,0))
         {}

         void clear()
         {
            n0_e  = std::make_pair(false,expression_node_ptr(0));
            n1_e  = std::make_pair(false,expression_node_ptr(0));
            n0_c  = std::make_pair(false,0);
            n1_c  = std::make_pair(false,0);
            cache = std::make_pair(0,0);
         }

         void free()
         {
            if (n0_e.first && n0_e.second)
            {
               n0_e.first = false;

               if (
                    !is_variable_node(n0_e.second) &&
                    !is_string_node  (n0_e.second)
                  )
               {
                  destroy_node(n0_e.second);
               }
            }

            if (n1_e.first && n1_e.second)
            {
               n1_e.first = false;

               if (
                    !is_variable_node(n1_e.second) &&
                    !is_string_node  (n1_e.second)
                  )
               {
                  destroy_node(n1_e.second);
               }
            }
         }

         bool const_range() const
         {
           return ( n0_c.first &&  n1_c.first) &&
                  (!n0_e.first && !n1_e.first);
         }

         bool var_range() const
         {
           return ( n0_e.first &&  n1_e.first) &&
                  (!n0_c.first && !n1_c.first);
         }

         bool operator() (std::size_t& r0, std::size_t& r1,
                          const std::size_t& size = std::numeric_limits<std::size_t>::max()) const
         {
            if (n0_c.first)
               r0 = n0_c.second;
            else if (n0_e.first)
            {
               r0 = static_cast<std::size_t>(details::numeric::to_int64(n0_e.second->value()));
            }
            else
               return false;

            if (n1_c.first)
               r1 = n1_c.second;
            else if (n1_e.first)
            {
               r1 = static_cast<std::size_t>(details::numeric::to_int64(n1_e.second->value()));
            }
            else
               return false;

            if (
                 (std::numeric_limits<std::size_t>::max() != size) &&
                 (std::numeric_limits<std::size_t>::max() == r1  )
               )
            {
               r1 = size - 1;
            }

            cache.first  = r0;
            cache.second = r1;

            #ifndef exprtk_enable_range_runtime_checks
            return (r0 <= r1);
            #else
            return range_runtime_check(r0, r1, size);
            #endif
         }

         inline std::size_t const_size() const
         {
            return (n1_c.second - n0_c.second + 1);
         }

         inline std::size_t cache_size() const
         {
            return (cache.second - cache.first + 1);
         }

         std::pair<bool,expression_node_ptr> n0_e;
         std::pair<bool,expression_node_ptr> n1_e;
         std::pair<bool,std::size_t        > n0_c;
         std::pair<bool,std::size_t        > n1_c;
         mutable cached_range_t             cache;

         #ifdef exprtk_enable_range_runtime_checks
         bool range_runtime_check(const std::size_t r0,
                                  const std::size_t r1,
                                  const std::size_t size) const
         {
            if (r0 >= size)
            {
               throw std::runtime_error("range error: (r0 < 0) || (r0 >= size)");
               return false;
            }

            if (r1 >= size)
            {
               throw std::runtime_error("range error: (r1 < 0) || (r1 >= size)");
               return false;
            }

            return (r0 <= r1);
         }
         #endif
      };

      template <typename T>
      class string_base_node;

      template <typename T>
      struct range_data_type
      {
         typedef range_pack<T> range_t;
         typedef string_base_node<T>* strbase_ptr_t;

         range_data_type()
         : range(0)
         , data (0)
         , size (0)
         , type_size(0)
         , str_node (0)
         {}

         range_t*      range;
         void*         data;
         std::size_t   size;
         std::size_t   type_size;
         strbase_ptr_t str_node;
      };

      template <typename T> class vector_node;

      template <typename T>
      class vector_interface
      {
      public:

         typedef vector_node<T>*   vector_node_ptr;
         typedef vec_data_store<T> vds_t;

         virtual ~vector_interface() {}

         virtual std::size_t size   () const = 0;

         virtual vector_node_ptr vec() const = 0;

         virtual vector_node_ptr vec()       = 0;

         virtual       vds_t& vds   ()       = 0;

         virtual const vds_t& vds   () const = 0;

         virtual bool side_effect   () const { return false; }
      };

      template <typename T>
      class vector_node exprtk_final
                        : public expression_node <T>
                        , public vector_interface<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef vector_holder<T>    vector_holder_t;
         typedef vector_node<T>*     vector_node_ptr;
         typedef vec_data_store<T>   vds_t;

         explicit vector_node(vector_holder_t* vh)
         : vector_holder_(vh)
         , vds_((*vector_holder_).size(),(*vector_holder_)[0])
         {
            vector_holder_->set_ref(&vds_.ref());
         }

         vector_node(const vds_t& vds, vector_holder_t* vh)
         : vector_holder_(vh)
         , vds_(vds)
         {}

         inline T value() const exprtk_override
         {
            return vds().data()[0];
         }

         vector_node_ptr vec() const exprtk_override
         {
            return const_cast<vector_node_ptr>(this);
         }

         vector_node_ptr vec() exprtk_override
         {
            return this;
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_vector;
         }

         std::size_t size() const exprtk_override
         {
            return vds().size();
         }

         vds_t& vds() exprtk_override
         {
            return vds_;
         }

         const vds_t& vds() const exprtk_override
         {
            return vds_;
         }

         inline vector_holder_t& vec_holder()
         {
            return (*vector_holder_);
         }

      private:

         vector_holder_t* vector_holder_;
         vds_t                      vds_;
      };

      template <typename T>
      class vector_elem_node exprtk_final
                             : public expression_node<T>,
                               public ivariable      <T>
      {
      public:

         typedef expression_node<T>*            expression_ptr;
         typedef vector_holder<T>               vector_holder_t;
         typedef vector_holder_t*               vector_holder_ptr;
         typedef std::pair<expression_ptr,bool> branch_t;

         vector_elem_node(expression_ptr index, vector_holder_ptr vec_holder)
         : vec_holder_(vec_holder)
         , vector_base_((*vec_holder)[0])
         {
            construct_branch_pair(index_, index);
         }

         inline T value() const exprtk_override
         {
            return *(vector_base_ + static_cast<std::size_t>(details::numeric::to_int64(index_.first->value())));
         }

         inline T& ref() exprtk_override
         {
            return *(vector_base_ + static_cast<std::size_t>(details::numeric::to_int64(index_.first->value())));
         }

         inline const T& ref() const exprtk_override
         {
            return *(vector_base_ + static_cast<std::size_t>(details::numeric::to_int64(index_.first->value())));
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_vecelem;
         }

         inline vector_holder_t& vec_holder()
         {
            return (*vec_holder_);
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(index_, node_delete_list);
         }

         std::size_t node_depth() const exprtk_override
         {
            return expression_node<T>::ndb_t::compute_node_depth(index_);
         }

      private:

         vector_holder_ptr vec_holder_;
         T* vector_base_;
         branch_t index_;
      };

      template <typename T>
      class rebasevector_elem_node exprtk_final
                                   : public expression_node<T>,
                                     public ivariable      <T>
      {
      public:

         typedef expression_node<T>*            expression_ptr;
         typedef vector_holder<T>               vector_holder_t;
         typedef vector_holder_t*               vector_holder_ptr;
         typedef vec_data_store<T>              vds_t;
         typedef std::pair<expression_ptr,bool> branch_t;

         rebasevector_elem_node(expression_ptr index, vector_holder_ptr vec_holder)
         : vector_holder_(vec_holder)
         , vds_((*vector_holder_).size(),(*vector_holder_)[0])
         {
            vector_holder_->set_ref(&vds_.ref());
            construct_branch_pair(index_, index);
         }

         inline T value() const exprtk_override
         {
            return *(vds_.data() + static_cast<std::size_t>(details::numeric::to_int64(index_.first->value())));
         }

         inline T& ref() exprtk_override
         {
            return *(vds_.data() + static_cast<std::size_t>(details::numeric::to_int64(index_.first->value())));
         }

         inline const T& ref() const exprtk_override
         {
            return *(vds_.data() + static_cast<std::size_t>(details::numeric::to_int64(index_.first->value())));
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_rbvecelem;
         }

         inline vector_holder_t& vec_holder()
         {
            return (*vector_holder_);
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(index_, node_delete_list);
         }

         std::size_t node_depth() const exprtk_override
         {
            return expression_node<T>::ndb_t::compute_node_depth(index_);
         }

      private:

         vector_holder_ptr vector_holder_;
         vds_t             vds_;
         branch_t          index_;
      };

      template <typename T>
      class rebasevector_celem_node exprtk_final
                                    : public expression_node<T>,
                                      public ivariable      <T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef vector_holder<T>    vector_holder_t;
         typedef vector_holder_t*    vector_holder_ptr;
         typedef vec_data_store<T>   vds_t;

         rebasevector_celem_node(const std::size_t index, vector_holder_ptr vec_holder)
         : index_(index)
         , vector_holder_(vec_holder)
         , vds_((*vector_holder_).size(),(*vector_holder_)[0])
         {
            vector_holder_->set_ref(&vds_.ref());
         }

         inline T value() const exprtk_override
         {
            return *(vds_.data() + index_);
         }

         inline T& ref() exprtk_override
         {
            return *(vds_.data() + index_);
         }

         inline const T& ref() const exprtk_override
         {
            return *(vds_.data() + index_);
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_rbveccelem;
         }

         inline vector_holder_t& vec_holder()
         {
            return (*vector_holder_);
         }

      private:

         const std::size_t index_;
         vector_holder_ptr vector_holder_;
         vds_t vds_;
      };

      template <typename T>
      class vector_assignment_node exprtk_final : public expression_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;

         vector_assignment_node(T* vector_base,
                                const std::size_t& size,
                                const std::vector<expression_ptr>& initialiser_list,
                                const bool single_value_initialse)
         : vector_base_(vector_base)
         , initialiser_list_(initialiser_list)
         , size_(size)
         , single_value_initialse_(single_value_initialse)
         {}

         inline T value() const exprtk_override
         {
            if (single_value_initialse_)
            {
               for (std::size_t i = 0; i < size_; ++i)
               {
                  *(vector_base_ + i) = initialiser_list_[0]->value();
               }
            }
            else
            {
               const std::size_t initialiser_list_size = initialiser_list_.size();

               for (std::size_t i = 0; i < initialiser_list_size; ++i)
               {
                  *(vector_base_ + i) = initialiser_list_[i]->value();
               }

               if (initialiser_list_size < size_)
               {
                  for (std::size_t i = initialiser_list_size; i < size_; ++i)
                  {
                     *(vector_base_ + i) = T(0);
                  }
               }
            }

            return *(vector_base_);
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_vecdefass;
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(initialiser_list_, node_delete_list);
         }

         std::size_t node_depth() const exprtk_override
         {
            return expression_node<T>::ndb_t::compute_node_depth(initialiser_list_);
         }

      private:

         vector_assignment_node(const vector_assignment_node<T>&) exprtk_delete;
         vector_assignment_node<T>& operator=(const vector_assignment_node<T>&) exprtk_delete;

         mutable T* vector_base_;
         std::vector<expression_ptr> initialiser_list_;
         const std::size_t size_;
         const bool single_value_initialse_;
      };

      template <typename T>
      class swap_node exprtk_final : public expression_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef variable_node<T>*   variable_node_ptr;

         swap_node(variable_node_ptr var0, variable_node_ptr var1)
         : var0_(var0)
         , var1_(var1)
         {}

         inline T value() const exprtk_override
         {
            std::swap(var0_->ref(),var1_->ref());
            return var1_->ref();
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_swap;
         }

      private:

         variable_node_ptr var0_;
         variable_node_ptr var1_;
      };

      template <typename T>
      class swap_generic_node exprtk_final : public binary_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef ivariable<T>*       ivariable_ptr;

         swap_generic_node(expression_ptr var0, expression_ptr var1)
         : binary_node<T>(details::e_swap, var0, var1)
         , var0_(dynamic_cast<ivariable_ptr>(var0))
         , var1_(dynamic_cast<ivariable_ptr>(var1))
         {}

         inline T value() const exprtk_override
         {
            std::swap(var0_->ref(),var1_->ref());
            return var1_->ref();
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_swap;
         }

      private:

         ivariable_ptr var0_;
         ivariable_ptr var1_;
      };

      template <typename T>
      class swap_vecvec_node exprtk_final
                             : public binary_node     <T>
                             , public vector_interface<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef vector_node    <T>* vector_node_ptr;
         typedef vec_data_store <T>  vds_t;

         swap_vecvec_node(expression_ptr branch0,
                          expression_ptr branch1)
         : binary_node<T>(details::e_swap, branch0, branch1)
         , vec0_node_ptr_(0)
         , vec1_node_ptr_(0)
         , vec_size_     (0)
         , initialised_  (false)
         {
            if (is_ivector_node(binary_node<T>::branch_[0].first))
            {
               vector_interface<T>* vi = reinterpret_cast<vector_interface<T>*>(0);

               if (0 != (vi = dynamic_cast<vector_interface<T>*>(binary_node<T>::branch_[0].first)))
               {
                  vec0_node_ptr_ = vi->vec();
                  vds()          = vi->vds();
               }
            }

            if (is_ivector_node(binary_node<T>::branch_[1].first))
            {
               vector_interface<T>* vi = reinterpret_cast<vector_interface<T>*>(0);

               if (0 != (vi = dynamic_cast<vector_interface<T>*>(binary_node<T>::branch_[1].first)))
               {
                  vec1_node_ptr_ = vi->vec();
               }
            }

            if (vec0_node_ptr_ && vec1_node_ptr_)
            {
               vec_size_ = std::min(vec0_node_ptr_->vds().size(),
                                    vec1_node_ptr_->vds().size());

               initialised_ = true;
            }

            assert(initialised_);
         }

         inline T value() const exprtk_override
         {
            if (initialised_)
            {
               assert(binary_node<T>::branch_[0].first);
               assert(binary_node<T>::branch_[1].first);

               binary_node<T>::branch_[0].first->value();
               binary_node<T>::branch_[1].first->value();

               T* vec0 = vec0_node_ptr_->vds().data();
               T* vec1 = vec1_node_ptr_->vds().data();

               for (std::size_t i = 0; i < vec_size_; ++i)
               {
                  std::swap(vec0[i],vec1[i]);
               }

               return vec1_node_ptr_->value();
            }
            else
               return std::numeric_limits<T>::quiet_NaN();
         }

         vector_node_ptr vec() const exprtk_override
         {
            return vec0_node_ptr_;
         }

         vector_node_ptr vec() exprtk_override
         {
            return vec0_node_ptr_;
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_vecvecswap;
         }

         std::size_t size() const exprtk_override
         {
            return vec_size_;
         }

         vds_t& vds() exprtk_override
         {
            return vds_;
         }

         const vds_t& vds() const exprtk_override
         {
            return vds_;
         }

      private:

         vector_node<T>* vec0_node_ptr_;
         vector_node<T>* vec1_node_ptr_;
         std::size_t     vec_size_;
         bool            initialised_;
         vds_t           vds_;
      };

      template <typename T, std::size_t N>
      inline T axn(const T a, const T x)
      {
         // a*x^n
         return a * exprtk::details::numeric::fast_exp<T,N>::result(x);
      }

      template <typename T, std::size_t N>
      inline T axnb(const T a, const T x, const T b)
      {
         // a*x^n+b
         return a * exprtk::details::numeric::fast_exp<T,N>::result(x) + b;
      }

      template <typename T>
      struct sf_base
      {
         typedef typename details::functor_t<T>::Type Type;
         typedef typename details::functor_t<T> functor_t;
         typedef typename functor_t::qfunc_t quaternary_functor_t;
         typedef typename functor_t::tfunc_t trinary_functor_t;
         typedef typename functor_t::bfunc_t binary_functor_t;
         typedef typename functor_t::ufunc_t unary_functor_t;
      };

      #define define_sfop3(NN, OP0, OP1)                 \
      template <typename T>                              \
      struct sf##NN##_op : public sf_base<T>             \
      {                                                  \
         typedef typename sf_base<T>::Type const Type;   \
         static inline T process(Type x, Type y, Type z) \
         {                                               \
            return (OP0);                                \
         }                                               \
         static inline std::string id()                  \
         {                                               \
            return (OP1);                                \
         }                                               \
      };                                                 \

      define_sfop3(00,(x + y) / z       ,"(t+t)/t")
      define_sfop3(01,(x + y) * z       ,"(t+t)*t")
      define_sfop3(02,(x + y) - z       ,"(t+t)-t")
      define_sfop3(03,(x + y) + z       ,"(t+t)+t")
      define_sfop3(04,(x - y) + z       ,"(t-t)+t")
      define_sfop3(05,(x - y) / z       ,"(t-t)/t")
      define_sfop3(06,(x - y) * z       ,"(t-t)*t")
      define_sfop3(07,(x * y) + z       ,"(t*t)+t")
      define_sfop3(08,(x * y) - z       ,"(t*t)-t")
      define_sfop3(09,(x * y) / z       ,"(t*t)/t")
      define_sfop3(10,(x * y) * z       ,"(t*t)*t")
      define_sfop3(11,(x / y) + z       ,"(t/t)+t")
      define_sfop3(12,(x / y) - z       ,"(t/t)-t")
      define_sfop3(13,(x / y) / z       ,"(t/t)/t")
      define_sfop3(14,(x / y) * z       ,"(t/t)*t")
      define_sfop3(15,x / (y + z)       ,"t/(t+t)")
      define_sfop3(16,x / (y - z)       ,"t/(t-t)")
      define_sfop3(17,x / (y * z)       ,"t/(t*t)")
      define_sfop3(18,x / (y / z)       ,"t/(t/t)")
      define_sfop3(19,x * (y + z)       ,"t*(t+t)")
      define_sfop3(20,x * (y - z)       ,"t*(t-t)")
      define_sfop3(21,x * (y * z)       ,"t*(t*t)")
      define_sfop3(22,x * (y / z)       ,"t*(t/t)")
      define_sfop3(23,x - (y + z)       ,"t-(t+t)")
      define_sfop3(24,x - (y - z)       ,"t-(t-t)")
      define_sfop3(25,x - (y / z)       ,"t-(t/t)")
      define_sfop3(26,x - (y * z)       ,"t-(t*t)")
      define_sfop3(27,x + (y * z)       ,"t+(t*t)")
      define_sfop3(28,x + (y / z)       ,"t+(t/t)")
      define_sfop3(29,x + (y + z)       ,"t+(t+t)")
      define_sfop3(30,x + (y - z)       ,"t+(t-t)")
      define_sfop3(31,(axnb<T,2>(x,y,z)),"       ")
      define_sfop3(32,(axnb<T,3>(x,y,z)),"       ")
      define_sfop3(33,(axnb<T,4>(x,y,z)),"       ")
      define_sfop3(34,(axnb<T,5>(x,y,z)),"       ")
      define_sfop3(35,(axnb<T,6>(x,y,z)),"       ")
      define_sfop3(36,(axnb<T,7>(x,y,z)),"       ")
      define_sfop3(37,(axnb<T,8>(x,y,z)),"       ")
      define_sfop3(38,(axnb<T,9>(x,y,z)),"       ")
      define_sfop3(39,x * numeric::log(y)   + z,"")
      define_sfop3(40,x * numeric::log(y)   - z,"")
      define_sfop3(41,x * numeric::log10(y) + z,"")
      define_sfop3(42,x * numeric::log10(y) - z,"")
      define_sfop3(43,x * numeric::sin(y) + z  ,"")
      define_sfop3(44,x * numeric::sin(y) - z  ,"")
      define_sfop3(45,x * numeric::cos(y) + z  ,"")
      define_sfop3(46,x * numeric::cos(y) - z  ,"")
      define_sfop3(47,details::is_true(x) ? y : z,"")

      #define define_sfop4(NN, OP0, OP1)                         \
      template <typename T>                                      \
      struct sf##NN##_op : public sf_base<T>                     \
      {                                                          \
         typedef typename sf_base<T>::Type const Type;           \
         static inline T process(Type x, Type y, Type z, Type w) \
         {                                                       \
            return (OP0);                                        \
         }                                                       \
         static inline std::string id()                          \
         {                                                       \
            return (OP1);                                        \
         }                                                       \
      };                                                         \

      define_sfop4(48,(x + ((y + z) / w)),"t+((t+t)/t)")
      define_sfop4(49,(x + ((y + z) * w)),"t+((t+t)*t)")
      define_sfop4(50,(x + ((y - z) / w)),"t+((t-t)/t)")
      define_sfop4(51,(x + ((y - z) * w)),"t+((t-t)*t)")
      define_sfop4(52,(x + ((y * z) / w)),"t+((t*t)/t)")
      define_sfop4(53,(x + ((y * z) * w)),"t+((t*t)*t)")
      define_sfop4(54,(x + ((y / z) + w)),"t+((t/t)+t)")
      define_sfop4(55,(x + ((y / z) / w)),"t+((t/t)/t)")
      define_sfop4(56,(x + ((y / z) * w)),"t+((t/t)*t)")
      define_sfop4(57,(x - ((y + z) / w)),"t-((t+t)/t)")
      define_sfop4(58,(x - ((y + z) * w)),"t-((t+t)*t)")
      define_sfop4(59,(x - ((y - z) / w)),"t-((t-t)/t)")
      define_sfop4(60,(x - ((y - z) * w)),"t-((t-t)*t)")
      define_sfop4(61,(x - ((y * z) / w)),"t-((t*t)/t)")
      define_sfop4(62,(x - ((y * z) * w)),"t-((t*t)*t)")
      define_sfop4(63,(x - ((y / z) / w)),"t-((t/t)/t)")
      define_sfop4(64,(x - ((y / z) * w)),"t-((t/t)*t)")
      define_sfop4(65,(((x + y) * z) - w),"((t+t)*t)-t")
      define_sfop4(66,(((x - y) * z) - w),"((t-t)*t)-t")
      define_sfop4(67,(((x * y) * z) - w),"((t*t)*t)-t")
      define_sfop4(68,(((x / y) * z) - w),"((t/t)*t)-t")
      define_sfop4(69,(((x + y) / z) - w),"((t+t)/t)-t")
      define_sfop4(70,(((x - y) / z) - w),"((t-t)/t)-t")
      define_sfop4(71,(((x * y) / z) - w),"((t*t)/t)-t")
      define_sfop4(72,(((x / y) / z) - w),"((t/t)/t)-t")
      define_sfop4(73,((x * y) + (z * w)),"(t*t)+(t*t)")
      define_sfop4(74,((x * y) - (z * w)),"(t*t)-(t*t)")
      define_sfop4(75,((x * y) + (z / w)),"(t*t)+(t/t)")
      define_sfop4(76,((x * y) - (z / w)),"(t*t)-(t/t)")
      define_sfop4(77,((x / y) + (z / w)),"(t/t)+(t/t)")
      define_sfop4(78,((x / y) - (z / w)),"(t/t)-(t/t)")
      define_sfop4(79,((x / y) - (z * w)),"(t/t)-(t*t)")
      define_sfop4(80,(x / (y + (z * w))),"t/(t+(t*t))")
      define_sfop4(81,(x / (y - (z * w))),"t/(t-(t*t))")
      define_sfop4(82,(x * (y + (z * w))),"t*(t+(t*t))")
      define_sfop4(83,(x * (y - (z * w))),"t*(t-(t*t))")

      define_sfop4(84,(axn<T,2>(x,y) + axn<T,2>(z,w)),"")
      define_sfop4(85,(axn<T,3>(x,y) + axn<T,3>(z,w)),"")
      define_sfop4(86,(axn<T,4>(x,y) + axn<T,4>(z,w)),"")
      define_sfop4(87,(axn<T,5>(x,y) + axn<T,5>(z,w)),"")
      define_sfop4(88,(axn<T,6>(x,y) + axn<T,6>(z,w)),"")
      define_sfop4(89,(axn<T,7>(x,y) + axn<T,7>(z,w)),"")
      define_sfop4(90,(axn<T,8>(x,y) + axn<T,8>(z,w)),"")
      define_sfop4(91,(axn<T,9>(x,y) + axn<T,9>(z,w)),"")
      define_sfop4(92,((details::is_true(x) && details::is_true(y)) ? z : w),"")
      define_sfop4(93,((details::is_true(x) || details::is_true(y)) ? z : w),"")
      define_sfop4(94,((x <  y) ? z : w),"")
      define_sfop4(95,((x <= y) ? z : w),"")
      define_sfop4(96,((x >  y) ? z : w),"")
      define_sfop4(97,((x >= y) ? z : w),"")
      define_sfop4(98,(details::is_true(numeric::equal(x,y)) ? z : w),"")
      define_sfop4(99,(x * numeric::sin(y) + z * numeric::cos(w)),"")

      define_sfop4(ext00,((x + y) - (z * w)),"(t+t)-(t*t)")
      define_sfop4(ext01,((x + y) - (z / w)),"(t+t)-(t/t)")
      define_sfop4(ext02,((x + y) + (z * w)),"(t+t)+(t*t)")
      define_sfop4(ext03,((x + y) + (z / w)),"(t+t)+(t/t)")
      define_sfop4(ext04,((x - y) + (z * w)),"(t-t)+(t*t)")
      define_sfop4(ext05,((x - y) + (z / w)),"(t-t)+(t/t)")
      define_sfop4(ext06,((x - y) - (z * w)),"(t-t)-(t*t)")
      define_sfop4(ext07,((x - y) - (z / w)),"(t-t)-(t/t)")
      define_sfop4(ext08,((x + y) - (z - w)),"(t+t)-(t-t)")
      define_sfop4(ext09,((x + y) + (z - w)),"(t+t)+(t-t)")
      define_sfop4(ext10,((x + y) + (z + w)),"(t+t)+(t+t)")
      define_sfop4(ext11,((x + y) * (z - w)),"(t+t)*(t-t)")
      define_sfop4(ext12,((x + y) / (z - w)),"(t+t)/(t-t)")
      define_sfop4(ext13,((x - y) - (z + w)),"(t-t)-(t+t)")
      define_sfop4(ext14,((x - y) + (z + w)),"(t-t)+(t+t)")
      define_sfop4(ext15,((x - y) * (z + w)),"(t-t)*(t+t)")
      define_sfop4(ext16,((x - y) / (z + w)),"(t-t)/(t+t)")
      define_sfop4(ext17,((x * y) - (z + w)),"(t*t)-(t+t)")
      define_sfop4(ext18,((x / y) - (z + w)),"(t/t)-(t+t)")
      define_sfop4(ext19,((x * y) + (z + w)),"(t*t)+(t+t)")
      define_sfop4(ext20,((x / y) + (z + w)),"(t/t)+(t+t)")
      define_sfop4(ext21,((x * y) + (z - w)),"(t*t)+(t-t)")
      define_sfop4(ext22,((x / y) + (z - w)),"(t/t)+(t-t)")
      define_sfop4(ext23,((x * y) - (z - w)),"(t*t)-(t-t)")
      define_sfop4(ext24,((x / y) - (z - w)),"(t/t)-(t-t)")
      define_sfop4(ext25,((x + y) * (z * w)),"(t+t)*(t*t)")
      define_sfop4(ext26,((x + y) * (z / w)),"(t+t)*(t/t)")
      define_sfop4(ext27,((x + y) / (z * w)),"(t+t)/(t*t)")
      define_sfop4(ext28,((x + y) / (z / w)),"(t+t)/(t/t)")
      define_sfop4(ext29,((x - y) / (z * w)),"(t-t)/(t*t)")
      define_sfop4(ext30,((x - y) / (z / w)),"(t-t)/(t/t)")
      define_sfop4(ext31,((x - y) * (z * w)),"(t-t)*(t*t)")
      define_sfop4(ext32,((x - y) * (z / w)),"(t-t)*(t/t)")
      define_sfop4(ext33,((x * y) * (z + w)),"(t*t)*(t+t)")
      define_sfop4(ext34,((x / y) * (z + w)),"(t/t)*(t+t)")
      define_sfop4(ext35,((x * y) / (z + w)),"(t*t)/(t+t)")
      define_sfop4(ext36,((x / y) / (z + w)),"(t/t)/(t+t)")
      define_sfop4(ext37,((x * y) / (z - w)),"(t*t)/(t-t)")
      define_sfop4(ext38,((x / y) / (z - w)),"(t/t)/(t-t)")
      define_sfop4(ext39,((x * y) * (z - w)),"(t*t)*(t-t)")
      define_sfop4(ext40,((x * y) / (z * w)),"(t*t)/(t*t)")
      define_sfop4(ext41,((x / y) * (z / w)),"(t/t)*(t/t)")
      define_sfop4(ext42,((x / y) * (z - w)),"(t/t)*(t-t)")
      define_sfop4(ext43,((x * y) * (z * w)),"(t*t)*(t*t)")
      define_sfop4(ext44,(x + (y * (z / w))),"t+(t*(t/t))")
      define_sfop4(ext45,(x - (y * (z / w))),"t-(t*(t/t))")
      define_sfop4(ext46,(x + (y / (z * w))),"t+(t/(t*t))")
      define_sfop4(ext47,(x - (y / (z * w))),"t-(t/(t*t))")
      define_sfop4(ext48,(((x - y) - z) * w),"((t-t)-t)*t")
      define_sfop4(ext49,(((x - y) - z) / w),"((t-t)-t)/t")
      define_sfop4(ext50,(((x - y) + z) * w),"((t-t)+t)*t")
      define_sfop4(ext51,(((x - y) + z) / w),"((t-t)+t)/t")
      define_sfop4(ext52,((x + (y - z)) * w),"(t+(t-t))*t")
      define_sfop4(ext53,((x + (y - z)) / w),"(t+(t-t))/t")
      define_sfop4(ext54,((x + y) / (z + w)),"(t+t)/(t+t)")
      define_sfop4(ext55,((x - y) / (z - w)),"(t-t)/(t-t)")
      define_sfop4(ext56,((x + y) * (z + w)),"(t+t)*(t+t)")
      define_sfop4(ext57,((x - y) * (z - w)),"(t-t)*(t-t)")
      define_sfop4(ext58,((x - y) + (z - w)),"(t-t)+(t-t)")
      define_sfop4(ext59,((x - y) - (z - w)),"(t-t)-(t-t)")
      define_sfop4(ext60,((x / y) + (z * w)),"(t/t)+(t*t)")
      define_sfop4(ext61,(((x * y) * z) / w),"((t*t)*t)/t")

      #undef define_sfop3
      #undef define_sfop4

      template <typename T, typename SpecialFunction>
      class sf3_node exprtk_final : public trinary_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;

         sf3_node(const operator_type& opr,
                  expression_ptr branch0,
                  expression_ptr branch1,
                  expression_ptr branch2)
         : trinary_node<T>(opr, branch0, branch1, branch2)
         {}

         inline T value() const exprtk_override
         {
            assert(trinary_node<T>::branch_[0].first);
            assert(trinary_node<T>::branch_[1].first);
            assert(trinary_node<T>::branch_[2].first);

            const T x = trinary_node<T>::branch_[0].first->value();
            const T y = trinary_node<T>::branch_[1].first->value();
            const T z = trinary_node<T>::branch_[2].first->value();

            return SpecialFunction::process(x, y, z);
         }
      };

      template <typename T, typename SpecialFunction>
      class sf4_node exprtk_final : public quaternary_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;

         sf4_node(const operator_type& opr,
                  expression_ptr branch0,
                  expression_ptr branch1,
                  expression_ptr branch2,
                  expression_ptr branch3)
         : quaternary_node<T>(opr, branch0, branch1, branch2, branch3)
         {}

         inline T value() const exprtk_override
         {
            assert(quaternary_node<T>::branch_[0].first);
            assert(quaternary_node<T>::branch_[1].first);
            assert(quaternary_node<T>::branch_[2].first);
            assert(quaternary_node<T>::branch_[3].first);

            const T x = quaternary_node<T>::branch_[0].first->value();
            const T y = quaternary_node<T>::branch_[1].first->value();
            const T z = quaternary_node<T>::branch_[2].first->value();
            const T w = quaternary_node<T>::branch_[3].first->value();

            return SpecialFunction::process(x, y, z, w);
         }
      };

      template <typename T, typename SpecialFunction>
      class sf3_var_node exprtk_final : public expression_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;

         sf3_var_node(const T& v0, const T& v1, const T& v2)
         : v0_(v0)
         , v1_(v1)
         , v2_(v2)
         {}

         inline T value() const exprtk_override
         {
            return SpecialFunction::process(v0_, v1_, v2_);
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_trinary;
         }

      private:

         sf3_var_node(const sf3_var_node<T,SpecialFunction>&) exprtk_delete;
         sf3_var_node<T,SpecialFunction>& operator=(const sf3_var_node<T,SpecialFunction>&) exprtk_delete;

         const T& v0_;
         const T& v1_;
         const T& v2_;
      };

      template <typename T, typename SpecialFunction>
      class sf4_var_node exprtk_final : public expression_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;

         sf4_var_node(const T& v0, const T& v1, const T& v2, const T& v3)
         : v0_(v0)
         , v1_(v1)
         , v2_(v2)
         , v3_(v3)
         {}

         inline T value() const exprtk_override
         {
            return SpecialFunction::process(v0_, v1_, v2_, v3_);
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_trinary;
         }

      private:

         sf4_var_node(const sf4_var_node<T,SpecialFunction>&) exprtk_delete;
         sf4_var_node<T,SpecialFunction>& operator=(const sf4_var_node<T,SpecialFunction>&) exprtk_delete;

         const T& v0_;
         const T& v1_;
         const T& v2_;
         const T& v3_;
      };

      template <typename T, typename VarArgFunction>
      class vararg_node exprtk_final : public expression_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef std::pair<expression_ptr,bool> branch_t;

         template <typename Allocator,
                   template <typename, typename> class Sequence>
         explicit vararg_node(const Sequence<expression_ptr,Allocator>& arg_list)
         {
            arg_list_.resize(arg_list.size());

            for (std::size_t i = 0; i < arg_list.size(); ++i)
            {
               if (arg_list[i])
               {
                  construct_branch_pair(arg_list_[i],arg_list[i]);
               }
               else
               {
                  arg_list_.clear();
                  return;
               }
            }
         }

         inline T value() const exprtk_override
         {
            return VarArgFunction::process(arg_list_);
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_vararg;
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(arg_list_, node_delete_list);
         }

         std::size_t node_depth() const exprtk_override
         {
            return expression_node<T>::ndb_t::compute_node_depth(arg_list_);
         }

      private:

         std::vector<branch_t> arg_list_;
      };

      template <typename T, typename VarArgFunction>
      class vararg_varnode exprtk_final : public expression_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;

         template <typename Allocator,
                   template <typename, typename> class Sequence>
         explicit vararg_varnode(const Sequence<expression_ptr,Allocator>& arg_list)
         {
            arg_list_.resize(arg_list.size());

            for (std::size_t i = 0; i < arg_list.size(); ++i)
            {
               if (arg_list[i] && is_variable_node(arg_list[i]))
               {
                  variable_node<T>* var_node_ptr = static_cast<variable_node<T>*>(arg_list[i]);
                  arg_list_[i] = (&var_node_ptr->ref());
               }
               else
               {
                  arg_list_.clear();
                  return;
               }
            }
         }

         inline T value() const exprtk_override
         {
            if (!arg_list_.empty())
               return VarArgFunction::process(arg_list_);
            else
               return std::numeric_limits<T>::quiet_NaN();
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_vararg;
         }

      private:

         std::vector<const T*> arg_list_;
      };

      template <typename T, typename VecFunction>
      class vectorize_node exprtk_final : public expression_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef std::pair<expression_ptr,bool> branch_t;

         explicit vectorize_node(const expression_ptr v)
         : ivec_ptr_(0)
         {
            construct_branch_pair(v_, v);

            if (is_ivector_node(v_.first))
            {
               ivec_ptr_ = dynamic_cast<vector_interface<T>*>(v_.first);
            }
            else
               ivec_ptr_ = 0;
         }

         inline T value() const exprtk_override
         {
            if (ivec_ptr_)
            {
               assert(v_.first);

               v_.first->value();

               return VecFunction::process(ivec_ptr_);
            }
            else
               return std::numeric_limits<T>::quiet_NaN();
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_vecfunc;
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(v_, node_delete_list);
         }

         std::size_t node_depth() const exprtk_override
         {
            return expression_node<T>::ndb_t::compute_node_depth(v_);
         }

      private:

         vector_interface<T>* ivec_ptr_;
         branch_t                    v_;
      };

      template <typename T>
      class assignment_node exprtk_final : public binary_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;

         assignment_node(const operator_type& opr,
                         expression_ptr branch0,
                         expression_ptr branch1)
         : binary_node<T>(opr, branch0, branch1)
         , var_node_ptr_(0)
         {
            if (is_variable_node(binary_node<T>::branch_[0].first))
            {
               var_node_ptr_ = static_cast<variable_node<T>*>(binary_node<T>::branch_[0].first);
            }
         }

         inline T value() const exprtk_override
         {
            if (var_node_ptr_)
            {
               assert(binary_node<T>::branch_[1].first);

               T& result = var_node_ptr_->ref();

               result = binary_node<T>::branch_[1].first->value();

               return result;
            }
            else
               return std::numeric_limits<T>::quiet_NaN();
         }

      private:

         variable_node<T>* var_node_ptr_;
      };

      template <typename T>
      class assignment_vec_elem_node exprtk_final : public binary_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;

         assignment_vec_elem_node(const operator_type& opr,
                                  expression_ptr branch0,
                                  expression_ptr branch1)
         : binary_node<T>(opr, branch0, branch1)
         , vec_node_ptr_(0)
         {
            if (is_vector_elem_node(binary_node<T>::branch_[0].first))
            {
               vec_node_ptr_ = static_cast<vector_elem_node<T>*>(binary_node<T>::branch_[0].first);
            }
         }

         inline T value() const exprtk_override
         {
            if (vec_node_ptr_)
            {
               assert(binary_node<T>::branch_[1].first);

               T& result = vec_node_ptr_->ref();

               result = binary_node<T>::branch_[1].first->value();

               return result;
            }
            else
               return std::numeric_limits<T>::quiet_NaN();
         }

      private:

         vector_elem_node<T>* vec_node_ptr_;
      };

      template <typename T>
      class assignment_rebasevec_elem_node exprtk_final : public binary_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;

         assignment_rebasevec_elem_node(const operator_type& opr,
                                        expression_ptr branch0,
                                        expression_ptr branch1)
         : binary_node<T>(opr, branch0, branch1)
         , rbvec_node_ptr_(0)
         {
            if (is_rebasevector_elem_node(binary_node<T>::branch_[0].first))
            {
               rbvec_node_ptr_ = static_cast<rebasevector_elem_node<T>*>(binary_node<T>::branch_[0].first);
            }
         }

         inline T value() const exprtk_override
         {
            if (rbvec_node_ptr_)
            {
               assert(binary_node<T>::branch_[1].first);

               T& result = rbvec_node_ptr_->ref();

               result = binary_node<T>::branch_[1].first->value();

               return result;
            }
            else
               return std::numeric_limits<T>::quiet_NaN();
         }

      private:

         rebasevector_elem_node<T>* rbvec_node_ptr_;
      };

      template <typename T>
      class assignment_rebasevec_celem_node exprtk_final : public binary_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;

         assignment_rebasevec_celem_node(const operator_type& opr,
                                         expression_ptr branch0,
                                         expression_ptr branch1)
         : binary_node<T>(opr, branch0, branch1)
         , rbvec_node_ptr_(0)
         {
            if (is_rebasevector_celem_node(binary_node<T>::branch_[0].first))
            {
               rbvec_node_ptr_ = static_cast<rebasevector_celem_node<T>*>(binary_node<T>::branch_[0].first);
            }
         }

         inline T value() const exprtk_override
         {
            if (rbvec_node_ptr_)
            {
               assert(binary_node<T>::branch_[1].first);

               T& result = rbvec_node_ptr_->ref();

               result = binary_node<T>::branch_[1].first->value();

               return result;
            }
            else
               return std::numeric_limits<T>::quiet_NaN();
         }

      private:

         rebasevector_celem_node<T>* rbvec_node_ptr_;
      };

      template <typename T>
      class assignment_vec_node exprtk_final
                                : public binary_node     <T>
                                , public vector_interface<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef vector_node<T>*     vector_node_ptr;
         typedef vec_data_store<T>   vds_t;

         assignment_vec_node(const operator_type& opr,
                             expression_ptr branch0,
                             expression_ptr branch1)
         : binary_node<T>(opr, branch0, branch1)
         , vec_node_ptr_(0)
         {
            if (is_vector_node(binary_node<T>::branch_[0].first))
            {
               vec_node_ptr_ = static_cast<vector_node<T>*>(binary_node<T>::branch_[0].first);
               vds()         = vec_node_ptr_->vds();
            }
         }

         inline T value() const exprtk_override
         {
            if (vec_node_ptr_)
            {
               assert(binary_node<T>::branch_[1].first);

               const T v = binary_node<T>::branch_[1].first->value();

               T* vec = vds().data();

               loop_unroll::details lud(size());
               const T* upper_bound = vec + lud.upper_bound;

               while (vec < upper_bound)
               {
                  #define exprtk_loop(N) \
                  vec[N] = v;            \

                  exprtk_loop( 0) exprtk_loop( 1)
                  exprtk_loop( 2) exprtk_loop( 3)

                  vec += lud.batch_size;
               }

               exprtk_disable_fallthrough_begin
               switch (lud.remainder)
               {
                  #define case_stmt(N) \
                  case N : *vec++ = v; \

                  case_stmt( 3) case_stmt( 2)
                  case_stmt( 1)
               }
               exprtk_disable_fallthrough_end

               #undef exprtk_loop
               #undef case_stmt

               return vec_node_ptr_->value();
            }
            else
               return std::numeric_limits<T>::quiet_NaN();
         }

         vector_node_ptr vec() const exprtk_override
         {
            return vec_node_ptr_;
         }

         vector_node_ptr vec() exprtk_override
         {
            return vec_node_ptr_;
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_vecvalass;
         }

         std::size_t size() const exprtk_override
         {
            return vds().size();
         }

         vds_t& vds() exprtk_override
         {
            return vds_;
         }

         const vds_t& vds() const exprtk_override
         {
            return vds_;
         }

      private:

         vector_node<T>* vec_node_ptr_;
         vds_t           vds_;
      };

      template <typename T>
      class assignment_vecvec_node exprtk_final
                                   : public binary_node     <T>
                                   , public vector_interface<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef vector_node<T>*     vector_node_ptr;
         typedef vec_data_store<T>   vds_t;

         assignment_vecvec_node(const operator_type& opr,
                                expression_ptr branch0,
                                expression_ptr branch1)
         : binary_node<T>(opr, branch0, branch1)
         , vec0_node_ptr_(0)
         , vec1_node_ptr_(0)
         , initialised_(false)
         , src_is_ivec_(false)
         {
            if (is_vector_node(binary_node<T>::branch_[0].first))
            {
               vec0_node_ptr_ = static_cast<vector_node<T>*>(binary_node<T>::branch_[0].first);
               vds()          = vec0_node_ptr_->vds();
            }

            if (is_vector_node(binary_node<T>::branch_[1].first))
            {
               vec1_node_ptr_ = static_cast<vector_node<T>*>(binary_node<T>::branch_[1].first);
               vds_t::match_sizes(vds(),vec1_node_ptr_->vds());
            }
            else if (is_ivector_node(binary_node<T>::branch_[1].first))
            {
               vector_interface<T>* vi = reinterpret_cast<vector_interface<T>*>(0);

               if (0 != (vi = dynamic_cast<vector_interface<T>*>(binary_node<T>::branch_[1].first)))
               {
                  vec1_node_ptr_ = vi->vec();

                  if (!vi->side_effect())
                  {
                     vi->vds()    = vds();
                     src_is_ivec_ = true;
                  }
                  else
                     vds_t::match_sizes(vds(),vi->vds());
               }
            }

            initialised_ = (vec0_node_ptr_ && vec1_node_ptr_);

            assert(initialised_);
         }

         inline T value() const exprtk_override
         {
            if (initialised_)
            {
               assert(binary_node<T>::branch_[1].first);

               binary_node<T>::branch_[1].first->value();

               if (src_is_ivec_)
                  return vec0_node_ptr_->value();

               T* vec0 = vec0_node_ptr_->vds().data();
               T* vec1 = vec1_node_ptr_->vds().data();

               loop_unroll::details lud(size());
               const T* upper_bound = vec0 + lud.upper_bound;

               while (vec0 < upper_bound)
               {
                  #define exprtk_loop(N) \
                  vec0[N] = vec1[N];     \

                  exprtk_loop( 0) exprtk_loop( 1)
                  exprtk_loop( 2) exprtk_loop( 3)

                  vec0 += lud.batch_size;
                  vec1 += lud.batch_size;
               }

               exprtk_disable_fallthrough_begin
               switch (lud.remainder)
               {
                  #define case_stmt(N)        \
                  case N : *vec0++ = *vec1++; \

                  case_stmt( 3) case_stmt( 2)
                  case_stmt( 1)
               }
               exprtk_disable_fallthrough_end

               #undef exprtk_loop
               #undef case_stmt

               return vec0_node_ptr_->value();
            }
            else
               return std::numeric_limits<T>::quiet_NaN();
         }

         vector_node_ptr vec() exprtk_override
         {
            return vec0_node_ptr_;
         }

         vector_node_ptr vec() const exprtk_override
         {
            return vec0_node_ptr_;
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_vecvecass;
         }

         std::size_t size() const exprtk_override
         {
            return vds().size();
         }

         vds_t& vds() exprtk_override
         {
            return vds_;
         }

         const vds_t& vds() const exprtk_override
         {
            return vds_;
         }

      private:

         vector_node<T>* vec0_node_ptr_;
         vector_node<T>* vec1_node_ptr_;
         bool            initialised_;
         bool            src_is_ivec_;
         vds_t           vds_;
      };

      template <typename T, typename Operation>
      class assignment_op_node exprtk_final : public binary_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;

         assignment_op_node(const operator_type& opr,
                            expression_ptr branch0,
                            expression_ptr branch1)
         : binary_node<T>(opr, branch0, branch1)
         , var_node_ptr_(0)
         {
            if (is_variable_node(binary_node<T>::branch_[0].first))
            {
               var_node_ptr_ = static_cast<variable_node<T>*>(binary_node<T>::branch_[0].first);
            }
         }

         inline T value() const exprtk_override
         {
            if (var_node_ptr_)
            {
               assert(binary_node<T>::branch_[1].first);

               T& v = var_node_ptr_->ref();
               v = Operation::process(v,binary_node<T>::branch_[1].first->value());

               return v;
            }
            else
               return std::numeric_limits<T>::quiet_NaN();
         }

      private:

         variable_node<T>* var_node_ptr_;
      };

      template <typename T, typename Operation>
      class assignment_vec_elem_op_node exprtk_final : public binary_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;

         assignment_vec_elem_op_node(const operator_type& opr,
                                     expression_ptr branch0,
                                     expression_ptr branch1)
         : binary_node<T>(opr, branch0, branch1)
         , vec_node_ptr_(0)
         {
            if (is_vector_elem_node(binary_node<T>::branch_[0].first))
            {
               vec_node_ptr_ = static_cast<vector_elem_node<T>*>(binary_node<T>::branch_[0].first);
            }
         }

         inline T value() const exprtk_override
         {
            if (vec_node_ptr_)
            {
               assert(binary_node<T>::branch_[1].first);

               T& v = vec_node_ptr_->ref();
                  v = Operation::process(v,binary_node<T>::branch_[1].first->value());

               return v;
            }
            else
               return std::numeric_limits<T>::quiet_NaN();
         }

      private:

         vector_elem_node<T>* vec_node_ptr_;
      };

      template <typename T, typename Operation>
      class assignment_rebasevec_elem_op_node exprtk_final : public binary_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;

         assignment_rebasevec_elem_op_node(const operator_type& opr,
                                           expression_ptr branch0,
                                           expression_ptr branch1)
         : binary_node<T>(opr, branch0, branch1)
         , rbvec_node_ptr_(0)
         {
            if (is_rebasevector_elem_node(binary_node<T>::branch_[0].first))
            {
               rbvec_node_ptr_ = static_cast<rebasevector_elem_node<T>*>(binary_node<T>::branch_[0].first);
            }
         }

         inline T value() const exprtk_override
         {
            if (rbvec_node_ptr_)
            {
               assert(binary_node<T>::branch_[1].first);

               T& v = rbvec_node_ptr_->ref();
                  v = Operation::process(v,binary_node<T>::branch_[1].first->value());

               return v;
            }
            else
               return std::numeric_limits<T>::quiet_NaN();
         }

      private:

         rebasevector_elem_node<T>* rbvec_node_ptr_;
      };

      template <typename T, typename Operation>
      class assignment_rebasevec_celem_op_node exprtk_final : public binary_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;

         assignment_rebasevec_celem_op_node(const operator_type& opr,
                                            expression_ptr branch0,
                                            expression_ptr branch1)
         : binary_node<T>(opr, branch0, branch1)
         , rbvec_node_ptr_(0)
         {
            if (is_rebasevector_celem_node(binary_node<T>::branch_[0].first))
            {
               rbvec_node_ptr_ = static_cast<rebasevector_celem_node<T>*>(binary_node<T>::branch_[0].first);
            }
         }

         inline T value() const exprtk_override
         {
            if (rbvec_node_ptr_)
            {
               assert(binary_node<T>::branch_[1].first);

               T& v = rbvec_node_ptr_->ref();
                  v = Operation::process(v,binary_node<T>::branch_[1].first->value());

               return v;
            }
            else
               return std::numeric_limits<T>::quiet_NaN();
         }

      private:

         rebasevector_celem_node<T>* rbvec_node_ptr_;
      };

      template <typename T, typename Operation>
      class assignment_vec_op_node exprtk_final
                                   : public binary_node     <T>
                                   , public vector_interface<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef vector_node<T>*     vector_node_ptr;
         typedef vec_data_store<T>   vds_t;

         assignment_vec_op_node(const operator_type& opr,
                                expression_ptr branch0,
                                expression_ptr branch1)
         : binary_node<T>(opr, branch0, branch1)
         , vec_node_ptr_(0)
         {
            if (is_vector_node(binary_node<T>::branch_[0].first))
            {
               vec_node_ptr_ = static_cast<vector_node<T>*>(binary_node<T>::branch_[0].first);
               vds()         = vec_node_ptr_->vds();
            }
         }

         inline T value() const exprtk_override
         {
            if (vec_node_ptr_)
            {
               assert(binary_node<T>::branch_[1].first);

               const T v = binary_node<T>::branch_[1].first->value();

               T* vec = vds().data();

               loop_unroll::details lud(size());
               const T* upper_bound = vec + lud.upper_bound;

               while (vec < upper_bound)
               {
                  #define exprtk_loop(N)       \
                  Operation::assign(vec[N],v); \

                  exprtk_loop( 0) exprtk_loop( 1)
                  exprtk_loop( 2) exprtk_loop( 3)

                  vec += lud.batch_size;
               }

               exprtk_disable_fallthrough_begin
               switch (lud.remainder)
               {
                  #define case_stmt(N)                  \
                  case N : Operation::assign(*vec++,v); \

                  case_stmt( 3) case_stmt( 2)
                  case_stmt( 1)
               }
               exprtk_disable_fallthrough_end

               #undef exprtk_loop
               #undef case_stmt

               return vec_node_ptr_->value();
            }
            else
               return std::numeric_limits<T>::quiet_NaN();
         }

         vector_node_ptr vec() const exprtk_override
         {
            return vec_node_ptr_;
         }

         vector_node_ptr vec() exprtk_override
         {
            return vec_node_ptr_;
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_vecopvalass;
         }

         std::size_t size() const exprtk_override
         {
            return vds().size();
         }

         vds_t& vds() exprtk_override
         {
            return vds_;
         }

         const vds_t& vds() const exprtk_override
         {
            return vds_;
         }

         bool side_effect() const exprtk_override
         {
            return true;
         }

      private:

         vector_node<T>* vec_node_ptr_;
         vds_t           vds_;
      };

      template <typename T, typename Operation>
      class assignment_vecvec_op_node exprtk_final
                                      : public binary_node     <T>
                                      , public vector_interface<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef vector_node<T>*     vector_node_ptr;
         typedef vec_data_store<T>   vds_t;

         assignment_vecvec_op_node(const operator_type& opr,
                                   expression_ptr branch0,
                                   expression_ptr branch1)
         : binary_node<T>(opr, branch0, branch1)
         , vec0_node_ptr_(0)
         , vec1_node_ptr_(0)
         , initialised_(false)
         {
            if (is_vector_node(binary_node<T>::branch_[0].first))
            {
               vec0_node_ptr_ = static_cast<vector_node<T>*>(binary_node<T>::branch_[0].first);
               vds()          = vec0_node_ptr_->vds();
            }

            if (is_vector_node(binary_node<T>::branch_[1].first))
            {
               vec1_node_ptr_ = static_cast<vector_node<T>*>(binary_node<T>::branch_[1].first);
               vec1_node_ptr_->vds() = vds();
            }
            else if (is_ivector_node(binary_node<T>::branch_[1].first))
            {
               vector_interface<T>* vi = reinterpret_cast<vector_interface<T>*>(0);

               if (0 != (vi = dynamic_cast<vector_interface<T>*>(binary_node<T>::branch_[1].first)))
               {
                  vec1_node_ptr_ = vi->vec();
                  vec1_node_ptr_->vds() = vds();
               }
               else
                  vds_t::match_sizes(vds(),vec1_node_ptr_->vds());
            }

            initialised_ = (vec0_node_ptr_ && vec1_node_ptr_);

            assert(initialised_);
         }

         inline T value() const exprtk_override
         {
            if (initialised_)
            {
               assert(binary_node<T>::branch_[0].first);
               assert(binary_node<T>::branch_[1].first);

               binary_node<T>::branch_[0].first->value();
               binary_node<T>::branch_[1].first->value();

                     T* vec0 = vec0_node_ptr_->vds().data();
               const T* vec1 = vec1_node_ptr_->vds().data();

               loop_unroll::details lud(size());
               const T* upper_bound = vec0 + lud.upper_bound;

               while (vec0 < upper_bound)
               {
                  #define exprtk_loop(N)                          \
                  vec0[N] = Operation::process(vec0[N], vec1[N]); \

                  exprtk_loop( 0) exprtk_loop( 1)
                  exprtk_loop( 2) exprtk_loop( 3)

                  vec0 += lud.batch_size;
                  vec1 += lud.batch_size;
               }

               int i = 0;

               exprtk_disable_fallthrough_begin
               switch (lud.remainder)
               {
                  #define case_stmt(N)                                              \
                  case N : { vec0[i] = Operation::process(vec0[i], vec1[i]); ++i; } \

                  case_stmt( 3) case_stmt( 2)
                  case_stmt( 1)
               }
               exprtk_disable_fallthrough_end

               #undef exprtk_loop
               #undef case_stmt

               return vec0_node_ptr_->value();
            }
            else
               return std::numeric_limits<T>::quiet_NaN();
         }

         vector_node_ptr vec() const exprtk_override
         {
            return vec0_node_ptr_;
         }

         vector_node_ptr vec() exprtk_override
         {
            return vec0_node_ptr_;
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_vecopvecass;
         }

         std::size_t size() const exprtk_override
         {
            return vds().size();
         }

         vds_t& vds() exprtk_override
         {
            return vds_;
         }

         const vds_t& vds() const exprtk_override
         {
            return vds_;
         }

         bool side_effect() const exprtk_override
         {
            return true;
         }

      private:

         vector_node<T>* vec0_node_ptr_;
         vector_node<T>* vec1_node_ptr_;
         bool            initialised_;
         vds_t           vds_;
      };

      template <typename T, typename Operation>
      class vec_binop_vecvec_node exprtk_final
                                  : public binary_node     <T>
                                  , public vector_interface<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef vector_node<T>*     vector_node_ptr;
         typedef vector_holder<T>*   vector_holder_ptr;
         typedef vec_data_store<T>   vds_t;

         vec_binop_vecvec_node(const operator_type& opr,
                               expression_ptr branch0,
                               expression_ptr branch1)
         : binary_node<T>(opr, branch0, branch1)
         , vec0_node_ptr_(0)
         , vec1_node_ptr_(0)
         , temp_         (0)
         , temp_vec_node_(0)
         , initialised_(false)
         {
            bool v0_is_ivec = false;
            bool v1_is_ivec = false;

            if (is_vector_node(binary_node<T>::branch_[0].first))
            {
               vec0_node_ptr_ = static_cast<vector_node_ptr>(binary_node<T>::branch_[0].first);
            }
            else if (is_ivector_node(binary_node<T>::branch_[0].first))
            {
               vector_interface<T>* vi = reinterpret_cast<vector_interface<T>*>(0);

               if (0 != (vi = dynamic_cast<vector_interface<T>*>(binary_node<T>::branch_[0].first)))
               {
                  vec0_node_ptr_ = vi->vec();
                  v0_is_ivec     = true;
               }
            }

            if (is_vector_node(binary_node<T>::branch_[1].first))
            {
               vec1_node_ptr_ = static_cast<vector_node_ptr>(binary_node<T>::branch_[1].first);
            }
            else if (is_ivector_node(binary_node<T>::branch_[1].first))
            {
               vector_interface<T>* vi = reinterpret_cast<vector_interface<T>*>(0);

               if (0 != (vi = dynamic_cast<vector_interface<T>*>(binary_node<T>::branch_[1].first)))
               {
                  vec1_node_ptr_ = vi->vec();
                  v1_is_ivec     = true;
               }
            }

            if (vec0_node_ptr_ && vec1_node_ptr_)
            {
               vector_holder<T>& vec0 = vec0_node_ptr_->vec_holder();
               vector_holder<T>& vec1 = vec1_node_ptr_->vec_holder();

               if (v0_is_ivec && (vec0.size() <= vec1.size()))
                  vds_ = vds_t(vec0_node_ptr_->vds());
               else if (v1_is_ivec && (vec1.size() <= vec0.size()))
                  vds_ = vds_t(vec1_node_ptr_->vds());
               else
                  vds_ = vds_t(std::min(vec0.size(),vec1.size()));

               temp_          = new vector_holder<T>(vds().data(),vds().size());
               temp_vec_node_ = new vector_node  <T>(vds(),temp_);

               initialised_ = true;
            }

            assert(initialised_);
         }

        ~vec_binop_vecvec_node()
         {
            delete temp_;
            delete temp_vec_node_;
         }

         inline T value() const exprtk_override
         {
            if (initialised_)
            {
               assert(binary_node<T>::branch_[0].first);
               assert(binary_node<T>::branch_[1].first);

               binary_node<T>::branch_[0].first->value();
               binary_node<T>::branch_[1].first->value();

               const T* vec0 = vec0_node_ptr_->vds().data();
               const T* vec1 = vec1_node_ptr_->vds().data();
                     T* vec2 = vds().data();

               loop_unroll::details lud(size());
               const T* upper_bound = vec2 + lud.upper_bound;

               while (vec2 < upper_bound)
               {
                  #define exprtk_loop(N)                          \
                  vec2[N] = Operation::process(vec0[N], vec1[N]); \

                  exprtk_loop( 0) exprtk_loop( 1)
                  exprtk_loop( 2) exprtk_loop( 3)

                  vec0 += lud.batch_size;
                  vec1 += lud.batch_size;
                  vec2 += lud.batch_size;
               }

               int i = 0;

               exprtk_disable_fallthrough_begin
               switch (lud.remainder)
               {
                  #define case_stmt(N)                                              \
                  case N : { vec2[i] = Operation::process(vec0[i], vec1[i]); ++i; } \

                  case_stmt( 3) case_stmt( 2)
                  case_stmt( 1)
               }
               exprtk_disable_fallthrough_end

               #undef exprtk_loop
               #undef case_stmt

               return (vds().data())[0];
            }
            else
               return std::numeric_limits<T>::quiet_NaN();
         }

         vector_node_ptr vec() const exprtk_override
         {
            return temp_vec_node_;
         }

         vector_node_ptr vec() exprtk_override
         {
            return temp_vec_node_;
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_vecvecarith;
         }

         std::size_t size() const exprtk_override
         {
            return vds_.size();
         }

         vds_t& vds() exprtk_override
         {
            return vds_;
         }

         const vds_t& vds() const exprtk_override
         {
            return vds_;
         }

      private:

         vector_node_ptr   vec0_node_ptr_;
         vector_node_ptr   vec1_node_ptr_;
         vector_holder_ptr temp_;
         vector_node_ptr   temp_vec_node_;
         bool              initialised_;
         vds_t             vds_;
      };

      template <typename T, typename Operation>
      class vec_binop_vecval_node exprtk_final
                                  : public binary_node     <T>
                                  , public vector_interface<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef vector_node<T>*     vector_node_ptr;
         typedef vector_holder<T>*   vector_holder_ptr;
         typedef vec_data_store<T>   vds_t;

         vec_binop_vecval_node(const operator_type& opr,
                               expression_ptr branch0,
                               expression_ptr branch1)
         : binary_node<T>(opr, branch0, branch1)
         , vec0_node_ptr_(0)
         , temp_         (0)
         , temp_vec_node_(0)
         {
            bool v0_is_ivec = false;

            if (is_vector_node(binary_node<T>::branch_[0].first))
            {
               vec0_node_ptr_ = static_cast<vector_node_ptr>(binary_node<T>::branch_[0].first);
            }
            else if (is_ivector_node(binary_node<T>::branch_[0].first))
            {
               vector_interface<T>* vi = reinterpret_cast<vector_interface<T>*>(0);

               if (0 != (vi = dynamic_cast<vector_interface<T>*>(binary_node<T>::branch_[0].first)))
               {
                  vec0_node_ptr_ = vi->vec();
                  v0_is_ivec     = true;
               }
            }

            if (vec0_node_ptr_)
            {
               if (v0_is_ivec)
                  vds() = vec0_node_ptr_->vds();
               else
                  vds() = vds_t(vec0_node_ptr_->size());

               temp_          = new vector_holder<T>(vds());
               temp_vec_node_ = new vector_node  <T>(vds(),temp_);
            }
         }

        ~vec_binop_vecval_node()
         {
            delete temp_;
            delete temp_vec_node_;
         }

         inline T value() const exprtk_override
         {
            if (vec0_node_ptr_)
            {
               assert(binary_node<T>::branch_[0].first);
               assert(binary_node<T>::branch_[1].first);

                           binary_node<T>::branch_[0].first->value();
               const T v = binary_node<T>::branch_[1].first->value();

               const T* vec0 = vec0_node_ptr_->vds().data();
                     T* vec1 = vds().data();

               loop_unroll::details lud(size());
               const T* upper_bound = vec0 + lud.upper_bound;

               while (vec0 < upper_bound)
               {
                  #define exprtk_loop(N)                    \
                  vec1[N] = Operation::process(vec0[N], v); \

                  exprtk_loop( 0) exprtk_loop( 1)
                  exprtk_loop( 2) exprtk_loop( 3)

                  vec0 += lud.batch_size;
                  vec1 += lud.batch_size;
               }

               int i = 0;

               exprtk_disable_fallthrough_begin
               switch (lud.remainder)
               {
                  #define case_stmt(N)                                        \
                  case N : { vec1[i] = Operation::process(vec0[i], v); ++i; } \

                  case_stmt( 3) case_stmt( 2)
                  case_stmt( 1)
               }
               exprtk_disable_fallthrough_end

               #undef exprtk_loop
               #undef case_stmt

               return (vds().data())[0];
            }
            else
               return std::numeric_limits<T>::quiet_NaN();
         }

         vector_node_ptr vec() const exprtk_override
         {
            return temp_vec_node_;
         }

         vector_node_ptr vec() exprtk_override
         {
            return temp_vec_node_;
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_vecvalarith;
         }

         std::size_t size() const exprtk_override
         {
            return vds().size();
         }

         vds_t& vds() exprtk_override
         {
            return vds_;
         }

         const vds_t& vds() const exprtk_override
         {
            return vds_;
         }

      private:

         vector_node_ptr   vec0_node_ptr_;
         vector_holder_ptr temp_;
         vector_node_ptr   temp_vec_node_;
         vds_t             vds_;
      };

      template <typename T, typename Operation>
      class vec_binop_valvec_node exprtk_final
                                  : public binary_node     <T>
                                  , public vector_interface<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef vector_node<T>*     vector_node_ptr;
         typedef vector_holder<T>*   vector_holder_ptr;
         typedef vec_data_store<T>   vds_t;

         vec_binop_valvec_node(const operator_type& opr,
                               expression_ptr branch0,
                               expression_ptr branch1)
         : binary_node<T>(opr, branch0, branch1)
         , vec1_node_ptr_(0)
         , temp_         (0)
         , temp_vec_node_(0)
         {
            bool v1_is_ivec = false;

            if (is_vector_node(binary_node<T>::branch_[1].first))
            {
               vec1_node_ptr_ = static_cast<vector_node_ptr>(binary_node<T>::branch_[1].first);
            }
            else if (is_ivector_node(binary_node<T>::branch_[1].first))
            {
               vector_interface<T>* vi = reinterpret_cast<vector_interface<T>*>(0);

               if (0 != (vi = dynamic_cast<vector_interface<T>*>(binary_node<T>::branch_[1].first)))
               {
                  vec1_node_ptr_ = vi->vec();
                  v1_is_ivec     = true;
               }
            }

            if (vec1_node_ptr_)
            {
               if (v1_is_ivec)
                  vds() = vec1_node_ptr_->vds();
               else
                  vds() = vds_t(vec1_node_ptr_->size());

               temp_          = new vector_holder<T>(vds());
               temp_vec_node_ = new vector_node  <T>(vds(),temp_);
            }
         }

        ~vec_binop_valvec_node()
         {
            delete temp_;
            delete temp_vec_node_;
         }

         inline T value() const exprtk_override
         {
            if (vec1_node_ptr_)
            {
               assert(binary_node<T>::branch_[0].first);
               assert(binary_node<T>::branch_[1].first);

               const T v = binary_node<T>::branch_[0].first->value();
                           binary_node<T>::branch_[1].first->value();

                     T* vec0 = vds().data();
               const T* vec1 = vec1_node_ptr_->vds().data();

               loop_unroll::details lud(size());
               const T* upper_bound = vec0 + lud.upper_bound;

               while (vec0 < upper_bound)
               {
                  #define exprtk_loop(N)                    \
                  vec0[N] = Operation::process(v, vec1[N]); \

                  exprtk_loop( 0) exprtk_loop( 1)
                  exprtk_loop( 2) exprtk_loop( 3)

                  vec0 += lud.batch_size;
                  vec1 += lud.batch_size;
               }

               int i = 0;

               exprtk_disable_fallthrough_begin
               switch (lud.remainder)
               {
                  #define case_stmt(N)                                        \
                  case N : { vec0[i] = Operation::process(v, vec1[i]); ++i; } \

                  case_stmt( 3) case_stmt( 2)
                  case_stmt( 1)
               }
               exprtk_disable_fallthrough_end

               #undef exprtk_loop
               #undef case_stmt

               return (vds().data())[0];
            }
            else
               return std::numeric_limits<T>::quiet_NaN();
         }

         vector_node_ptr vec() const exprtk_override
         {
            return temp_vec_node_;
         }

         vector_node_ptr vec() exprtk_override
         {
            return temp_vec_node_;
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_vecvalarith;
         }

         std::size_t size() const exprtk_override
         {
            return vds().size();
         }

         vds_t& vds() exprtk_override
         {
            return vds_;
         }

         const vds_t& vds() const exprtk_override
         {
            return vds_;
         }

      private:

         vector_node_ptr   vec1_node_ptr_;
         vector_holder_ptr temp_;
         vector_node_ptr   temp_vec_node_;
         vds_t             vds_;
      };

      template <typename T, typename Operation>
      class unary_vector_node exprtk_final
                              : public unary_node      <T>
                              , public vector_interface<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef vector_node<T>*     vector_node_ptr;
         typedef vector_holder<T>*   vector_holder_ptr;
         typedef vec_data_store<T>   vds_t;

         unary_vector_node(const operator_type& opr, expression_ptr branch0)
         : unary_node<T>(opr, branch0)
         , vec0_node_ptr_(0)
         , temp_         (0)
         , temp_vec_node_(0)
         {
            bool vec0_is_ivec = false;

            if (is_vector_node(unary_node<T>::branch_.first))
            {
               vec0_node_ptr_ = static_cast<vector_node_ptr>(unary_node<T>::branch_.first);
            }
            else if (is_ivector_node(unary_node<T>::branch_.first))
            {
               vector_interface<T>* vi = reinterpret_cast<vector_interface<T>*>(0);

               if (0 != (vi = dynamic_cast<vector_interface<T>*>(unary_node<T>::branch_.first)))
               {
                  vec0_node_ptr_ = vi->vec();
                  vec0_is_ivec   = true;
               }
            }

            if (vec0_node_ptr_)
            {
               if (vec0_is_ivec)
                  vds_ = vec0_node_ptr_->vds();
               else
                  vds_ = vds_t(vec0_node_ptr_->size());

               temp_          = new vector_holder<T>(vds());
               temp_vec_node_ = new vector_node  <T>(vds(),temp_);
            }
         }

        ~unary_vector_node()
         {
            delete temp_;
            delete temp_vec_node_;
         }

         inline T value() const exprtk_override
         {
            assert(unary_node<T>::branch_.first);

            unary_node<T>::branch_.first->value();

            if (vec0_node_ptr_)
            {
               const T* vec0 = vec0_node_ptr_->vds().data();
                     T* vec1 = vds().data();

               loop_unroll::details lud(size());
               const T* upper_bound = vec0 + lud.upper_bound;

               while (vec0 < upper_bound)
               {
                  #define exprtk_loop(N)                 \
                  vec1[N] = Operation::process(vec0[N]); \

                  exprtk_loop( 0) exprtk_loop( 1)
                  exprtk_loop( 2) exprtk_loop( 3)

                  vec0 += lud.batch_size;
                  vec1 += lud.batch_size;
               }

               int i = 0;

               exprtk_disable_fallthrough_begin
               switch (lud.remainder)
               {
                  #define case_stmt(N)                                     \
                  case N : { vec1[i] = Operation::process(vec0[i]); ++i; } \

                  case_stmt( 3) case_stmt( 2)
                  case_stmt( 1)
               }
               exprtk_disable_fallthrough_end

               #undef exprtk_loop
               #undef case_stmt

               return (vds().data())[0];
            }
            else
               return std::numeric_limits<T>::quiet_NaN();
         }

         vector_node_ptr vec() const exprtk_override
         {
            return temp_vec_node_;
         }

         vector_node_ptr vec() exprtk_override
         {
            return temp_vec_node_;
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_vecunaryop;
         }

         std::size_t size() const exprtk_override
         {
            return vds().size();
         }

         vds_t& vds() exprtk_override
         {
            return vds_;
         }

         const vds_t& vds() const exprtk_override
         {
            return vds_;
         }

      private:

         vector_node_ptr   vec0_node_ptr_;
         vector_holder_ptr temp_;
         vector_node_ptr   temp_vec_node_;
         vds_t             vds_;
      };

      template <typename T>
      class conditional_vector_node exprtk_final
                                    : public expression_node <T>
                                    , public vector_interface<T>
      {
      public:

         typedef expression_node <T>* expression_ptr;
         typedef vector_interface<T>* vec_interface_ptr;
         typedef vector_node     <T>* vector_node_ptr;
         typedef vector_holder   <T>* vector_holder_ptr;
         typedef vec_data_store  <T>  vds_t;
         typedef std::pair<expression_ptr,bool> branch_t;

         conditional_vector_node(expression_ptr condition,
                                 expression_ptr consequent,
                                 expression_ptr alternative)
         : consequent_node_ptr_ (0)
         , alternative_node_ptr_(0)
         , temp_vec_node_       (0)
         , temp_                (0)
         , vec_size_            (0)
         , initialised_         (false)
         {
            construct_branch_pair(condition_  , condition  );
            construct_branch_pair(consequent_ , consequent );
            construct_branch_pair(alternative_, alternative);

            if (details::is_ivector_node(consequent_.first))
            {
               vec_interface_ptr ivec_ptr = dynamic_cast<vec_interface_ptr>(consequent_.first);

               if (0 != ivec_ptr)
               {
                  consequent_node_ptr_ = ivec_ptr->vec();
               }
            }

            if (details::is_ivector_node(alternative_.first))
            {
               vec_interface_ptr ivec_ptr = dynamic_cast<vec_interface_ptr>(alternative_.first);

               if (0 != ivec_ptr)
               {
                  alternative_node_ptr_ = ivec_ptr->vec();
               }
            }

            if (consequent_node_ptr_ && alternative_node_ptr_)
            {
               vec_size_ = std::min(consequent_node_ptr_ ->vds().size(),
                                    alternative_node_ptr_->vds().size());

               vds_           = vds_t(vec_size_);
               temp_          = new vector_holder<T>(vds_);
               temp_vec_node_ = new vector_node  <T>(vds(),temp_);

               initialised_ = true;
            }

            assert(initialised_ && (vec_size_ > 0));
         }

        ~conditional_vector_node()
         {
            delete temp_;
            delete temp_vec_node_;
         }

         inline T value() const exprtk_override
         {
            if (initialised_)
            {
               assert(condition_  .first);
               assert(consequent_ .first);
               assert(alternative_.first);

               T result = T(0);
               T* source_vector = 0;
               T* result_vector = vds().data();

               if (is_true(condition_))
               {
                  result        = consequent_.first->value();
                  source_vector = consequent_node_ptr_->vds().data();
               }
               else
               {
                  result        = alternative_.first->value();
                  source_vector = alternative_node_ptr_->vds().data();
               }

               for (std::size_t i = 0; i < vec_size_; ++i)
               {
                  result_vector[i] = source_vector[i];
               }

               return result;
            }

            return std::numeric_limits<T>::quiet_NaN();
         }

         vector_node_ptr vec() const exprtk_override
         {
            return temp_vec_node_;
         }

         vector_node_ptr vec() exprtk_override
         {
            return temp_vec_node_;
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_vecondition;
         }

         std::size_t size() const exprtk_override
         {
            return vec_size_;
         }

         vds_t& vds() exprtk_override
         {
            return vds_;
         }

         const vds_t& vds() const exprtk_override
         {
            return vds_;
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(condition_   , node_delete_list);
            expression_node<T>::ndb_t::collect(consequent_  , node_delete_list);
            expression_node<T>::ndb_t::collect(alternative_ , node_delete_list);
         }

         std::size_t node_depth() const exprtk_override
         {
            return expression_node<T>::ndb_t::compute_node_depth
               (condition_, consequent_, alternative_);
         }

      private:

         branch_t condition_;
         branch_t consequent_;
         branch_t alternative_;
         vector_node_ptr   consequent_node_ptr_;
         vector_node_ptr   alternative_node_ptr_;
         vector_node_ptr   temp_vec_node_;
         vector_holder_ptr temp_;
         vds_t vds_;
         std::size_t vec_size_;
         bool        initialised_;
      };

      template <typename T>
      class scand_node exprtk_final : public binary_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;

         scand_node(const operator_type& opr,
                    expression_ptr branch0,
                    expression_ptr branch1)
         : binary_node<T>(opr, branch0, branch1)
         {}

         inline T value() const exprtk_override
         {
            assert(binary_node<T>::branch_[0].first);
            assert(binary_node<T>::branch_[1].first);

            return (
                     std::not_equal_to<T>()
                        (T(0),binary_node<T>::branch_[0].first->value()) &&
                     std::not_equal_to<T>()
                        (T(0),binary_node<T>::branch_[1].first->value())
                   ) ? T(1) : T(0);
         }
      };

      template <typename T>
      class scor_node exprtk_final : public binary_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;

         scor_node(const operator_type& opr,
                   expression_ptr branch0,
                   expression_ptr branch1)
         : binary_node<T>(opr, branch0, branch1)
         {}

         inline T value() const exprtk_override
         {
            assert(binary_node<T>::branch_[0].first);
            assert(binary_node<T>::branch_[1].first);

            return (
                     std::not_equal_to<T>()
                        (T(0),binary_node<T>::branch_[0].first->value()) ||
                     std::not_equal_to<T>()
                        (T(0),binary_node<T>::branch_[1].first->value())
                   ) ? T(1) : T(0);
         }
      };

      template <typename T, typename IFunction, std::size_t N>
      class function_N_node exprtk_final : public expression_node<T>
      {
      public:

         // Function of N paramters.
         typedef expression_node<T>* expression_ptr;
         typedef std::pair<expression_ptr,bool> branch_t;
         typedef IFunction ifunction;

         explicit function_N_node(ifunction* func)
         : function_((N == func->param_count) ? func : reinterpret_cast<ifunction*>(0))
         , parameter_count_(func->param_count)
         {}

         template <std::size_t NumBranches>
         bool init_branches(expression_ptr (&b)[NumBranches])
         {
            // Needed for incompetent and broken msvc compiler versions
            #ifdef _MSC_VER
             #pragma warning(push)
             #pragma warning(disable: 4127)
            #endif
            if (N != NumBranches)
               return false;
            else
            {
               for (std::size_t i = 0; i < NumBranches; ++i)
               {
                  if (b[i])
                     branch_[i] = std::make_pair(b[i],branch_deletable(b[i]));
                  else
                     return false;
               }
               return true;
            }
            #ifdef _MSC_VER
             #pragma warning(pop)
            #endif
         }

         inline bool operator <(const function_N_node<T,IFunction,N>& fn) const
         {
            return this < (&fn);
         }

         inline T value() const exprtk_override
         {
            // Needed for incompetent and broken msvc compiler versions
            #ifdef _MSC_VER
             #pragma warning(push)
             #pragma warning(disable: 4127)
            #endif
            if ((0 == function_) || (0 == N))
               return std::numeric_limits<T>::quiet_NaN();
            else
            {
               T v[N];
               evaluate_branches<T,N>::execute(v,branch_);
               return invoke<T,N>::execute(*function_,v);
            }
            #ifdef _MSC_VER
             #pragma warning(pop)
            #endif
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_function;
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(branch_, node_delete_list);
         }

         std::size_t node_depth() const exprtk_override
         {
            return expression_node<T>::ndb_t::template compute_node_depth<N>(branch_);
         }

         template <typename T_, std::size_t BranchCount>
         struct evaluate_branches
         {
            static inline void execute(T_ (&v)[BranchCount], const branch_t (&b)[BranchCount])
            {
               for (std::size_t i = 0; i < BranchCount; ++i)
               {
                  v[i] = b[i].first->value();
               }
            }
         };

         template <typename T_>
         struct evaluate_branches <T_,5>
         {
            static inline void execute(T_ (&v)[5], const branch_t (&b)[5])
            {
               v[0] = b[0].first->value();
               v[1] = b[1].first->value();
               v[2] = b[2].first->value();
               v[3] = b[3].first->value();
               v[4] = b[4].first->value();
            }
         };

         template <typename T_>
         struct evaluate_branches <T_,4>
         {
            static inline void execute(T_ (&v)[4], const branch_t (&b)[4])
            {
               v[0] = b[0].first->value();
               v[1] = b[1].first->value();
               v[2] = b[2].first->value();
               v[3] = b[3].first->value();
            }
         };

         template <typename T_>
         struct evaluate_branches <T_,3>
         {
            static inline void execute(T_ (&v)[3], const branch_t (&b)[3])
            {
               v[0] = b[0].first->value();
               v[1] = b[1].first->value();
               v[2] = b[2].first->value();
            }
         };

         template <typename T_>
         struct evaluate_branches <T_,2>
         {
            static inline void execute(T_ (&v)[2], const branch_t (&b)[2])
            {
               v[0] = b[0].first->value();
               v[1] = b[1].first->value();
            }
         };

         template <typename T_>
         struct evaluate_branches <T_,1>
         {
            static inline void execute(T_ (&v)[1], const branch_t (&b)[1])
            {
               v[0] = b[0].first->value();
            }
         };

         template <typename T_, std::size_t ParamCount>
         struct invoke { static inline T execute(ifunction&, branch_t (&)[ParamCount]) { return std::numeric_limits<T_>::quiet_NaN(); } };

         template <typename T_>
         struct invoke<T_,20>
         {
            static inline T_ execute(ifunction& f, T_ (&v)[20])
            { return f(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12],v[13],v[14],v[15],v[16],v[17],v[18],v[19]); }
         };

         template <typename T_>
         struct invoke<T_,19>
         {
            static inline T_ execute(ifunction& f, T_ (&v)[19])
            { return f(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12],v[13],v[14],v[15],v[16],v[17],v[18]); }
         };

         template <typename T_>
         struct invoke<T_,18>
         {
            static inline T_ execute(ifunction& f, T_ (&v)[18])
            { return f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15], v[16], v[17]); }
         };

         template <typename T_>
         struct invoke<T_,17>
         {
            static inline T_ execute(ifunction& f, T_ (&v)[17])
            { return f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15], v[16]); }
         };

         template <typename T_>
         struct invoke<T_,16>
         {
            static inline T_ execute(ifunction& f, T_ (&v)[16])
            { return f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]); }
         };

         template <typename T_>
         struct invoke<T_,15>
         {
            static inline T_ execute(ifunction& f, T_ (&v)[15])
            { return f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14]); }
         };

         template <typename T_>
         struct invoke<T_,14>
         {
            static inline T_ execute(ifunction& f, T_ (&v)[14])
            { return f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13]); }
         };

         template <typename T_>
         struct invoke<T_,13>
         {
            static inline T_ execute(ifunction& f, T_ (&v)[13])
            { return f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12]); }
         };

         template <typename T_>
         struct invoke<T_,12>
         {
            static inline T_ execute(ifunction& f, T_ (&v)[12])
            { return f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11]); }
         };

         template <typename T_>
         struct invoke<T_,11>
         {
            static inline T_ execute(ifunction& f, T_ (&v)[11])
            { return f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10]); }
         };

         template <typename T_>
         struct invoke<T_,10>
         {
            static inline T_ execute(ifunction& f, T_ (&v)[10])
            { return f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9]); }
         };

         template <typename T_>
         struct invoke<T_,9>
         {
            static inline T_ execute(ifunction& f, T_ (&v)[9])
            { return f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]); }
         };

         template <typename T_>
         struct invoke<T_,8>
         {
            static inline T_ execute(ifunction& f, T_ (&v)[8])
            { return f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]); }
         };

         template <typename T_>
         struct invoke<T_,7>
         {
            static inline T_ execute(ifunction& f, T_ (&v)[7])
            { return f(v[0], v[1], v[2], v[3], v[4], v[5], v[6]); }
         };

         template <typename T_>
         struct invoke<T_,6>
         {
            static inline T_ execute(ifunction& f, T_ (&v)[6])
            { return f(v[0], v[1], v[2], v[3], v[4], v[5]); }
         };

         template <typename T_>
         struct invoke<T_,5>
         {
            static inline T_ execute(ifunction& f, T_ (&v)[5])
            { return f(v[0], v[1], v[2], v[3], v[4]); }
         };

         template <typename T_>
         struct invoke<T_,4>
         {
            static inline T_ execute(ifunction& f, T_ (&v)[4])
            { return f(v[0], v[1], v[2], v[3]); }
         };

         template <typename T_>
         struct invoke<T_,3>
         {
            static inline T_ execute(ifunction& f, T_ (&v)[3])
            { return f(v[0], v[1], v[2]); }
         };

         template <typename T_>
         struct invoke<T_,2>
         {
            static inline T_ execute(ifunction& f, T_ (&v)[2])
            { return f(v[0], v[1]); }
         };

         template <typename T_>
         struct invoke<T_,1>
         {
            static inline T_ execute(ifunction& f, T_ (&v)[1])
            { return f(v[0]); }
         };

      private:

         ifunction*  function_;
         std::size_t parameter_count_;
         branch_t    branch_[N];
      };

      template <typename T, typename IFunction>
      class function_N_node<T,IFunction,0> exprtk_final : public expression_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef IFunction ifunction;

         explicit function_N_node(ifunction* func)
         : function_((0 == func->param_count) ? func : reinterpret_cast<ifunction*>(0))
         {}

         inline bool operator <(const function_N_node<T,IFunction,0>& fn) const
         {
            return this < (&fn);
         }

         inline T value() const exprtk_override
         {
            if (function_)
               return (*function_)();
            else
               return std::numeric_limits<T>::quiet_NaN();
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_function;
         }

      private:

         ifunction* function_;
      };

      template <typename T, typename VarArgFunction>
      class vararg_function_node exprtk_final : public expression_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;

         vararg_function_node(VarArgFunction*  func,
                              const std::vector<expression_ptr>& arg_list)
         : function_(func)
         , arg_list_(arg_list)
         {
            value_list_.resize(arg_list.size(),std::numeric_limits<T>::quiet_NaN());
         }

         inline bool operator <(const vararg_function_node<T,VarArgFunction>& fn) const
         {
            return this < (&fn);
         }

         inline T value() const exprtk_override
         {
            if (function_)
            {
               populate_value_list();
               return (*function_)(value_list_);
            }
            else
               return std::numeric_limits<T>::quiet_NaN();
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_vafunction;
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            for (std::size_t i = 0; i < arg_list_.size(); ++i)
            {
               if (arg_list_[i] && !details::is_variable_node(arg_list_[i]))
               {
                  node_delete_list.push_back(&arg_list_[i]);
               }
            }
         }

         std::size_t node_depth() const exprtk_override
         {
            return expression_node<T>::ndb_t::compute_node_depth(arg_list_);
         }

      private:

         inline void populate_value_list() const
         {
            for (std::size_t i = 0; i < arg_list_.size(); ++i)
            {
               value_list_[i] = arg_list_[i]->value();
            }
         }

         VarArgFunction* function_;
         std::vector<expression_ptr> arg_list_;
         mutable std::vector<T> value_list_;
      };

      template <typename T, typename GenericFunction>
      class generic_function_node : public expression_node<T>
      {
      public:

         typedef type_store<T>       type_store_t;
         typedef expression_node<T>* expression_ptr;
         typedef variable_node<T>    variable_node_t;
         typedef vector_node<T>      vector_node_t;
         typedef variable_node_t*    variable_node_ptr_t;
         typedef vector_node_t*      vector_node_ptr_t;
         typedef range_interface<T>  range_interface_t;
         typedef range_data_type<T>  range_data_type_t;
         typedef typename range_interface<T>::range_t range_t;

         typedef std::pair<expression_ptr,bool> branch_t;
         typedef std::pair<void*,std::size_t>   void_t;

         typedef std::vector<T>                 tmp_vs_t;
         typedef std::vector<type_store_t>      typestore_list_t;
         typedef std::vector<range_data_type_t> range_list_t;

         explicit generic_function_node(const std::vector<expression_ptr>& arg_list,
                                        GenericFunction* func = reinterpret_cast<GenericFunction*>(0))
         : function_(func)
         , arg_list_(arg_list)
         {}

         virtual ~generic_function_node() {}

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(branch_, node_delete_list);
         }

         std::size_t node_depth() const exprtk_override exprtk_final
         {
            return expression_node<T>::ndb_t::compute_node_depth(branch_);
         }

         virtual bool init_branches()
         {
            expr_as_vec1_store_.resize(arg_list_.size(),T(0)               );
            typestore_list_    .resize(arg_list_.size(),type_store_t()     );
            range_list_        .resize(arg_list_.size(),range_data_type_t());
            branch_            .resize(arg_list_.size(),branch_t(reinterpret_cast<expression_ptr>(0),false));

            for (std::size_t i = 0; i < arg_list_.size(); ++i)
            {
               type_store_t& ts = typestore_list_[i];

               if (0 == arg_list_[i])
                  return false;
               else if (is_ivector_node(arg_list_[i]))
               {
                  vector_interface<T>* vi = reinterpret_cast<vector_interface<T>*>(0);

                  if (0 == (vi = dynamic_cast<vector_interface<T>*>(arg_list_[i])))
                     return false;

                  ts.size = vi->size();
                  ts.data = vi->vds().data();
                  ts.type = type_store_t::e_vector;
                  vi->vec()->vec_holder().set_ref(&ts.vec_data);
               }
               else if (is_variable_node(arg_list_[i]))
               {
                  variable_node_ptr_t var = variable_node_ptr_t(0);

                  if (0 == (var = dynamic_cast<variable_node_ptr_t>(arg_list_[i])))
                     return false;

                  ts.size = 1;
                  ts.data = &var->ref();
                  ts.type = type_store_t::e_scalar;
               }
               else
               {
                  ts.size = 1;
                  ts.data = reinterpret_cast<void*>(&expr_as_vec1_store_[i]);
                  ts.type = type_store_t::e_scalar;
               }

               branch_[i] = std::make_pair(arg_list_[i],branch_deletable(arg_list_[i]));
            }

            return true;
         }

         inline bool operator <(const generic_function_node<T,GenericFunction>& fn) const
         {
            return this < (&fn);
         }

         inline T value() const exprtk_override
         {
            if (function_)
            {
               if (populate_value_list())
               {
                  typedef typename GenericFunction::parameter_list_t parameter_list_t;

                  return (*function_)(parameter_list_t(typestore_list_));
               }
            }

            return std::numeric_limits<T>::quiet_NaN();
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_genfunction;
         }

      protected:

         inline virtual bool populate_value_list() const
         {
            for (std::size_t i = 0; i < branch_.size(); ++i)
            {
               expr_as_vec1_store_[i] = branch_[i].first->value();
            }

            for (std::size_t i = 0; i < branch_.size(); ++i)
            {
               range_data_type_t& rdt = range_list_[i];

               if (rdt.range)
               {
                  const range_t& rp = (*rdt.range);
                  std::size_t r0    = 0;
                  std::size_t r1    = 0;

                  if (rp(r0, r1, rdt.size))
                  {
                     type_store_t& ts = typestore_list_[i];

                     ts.size = rp.cache_size();
					 ts.data = static_cast<char_ptr>(rdt.data) + (rp.cache.first * rdt.type_size);
                  }
                  else
                     return false;
               }
            }

            return true;
         }

         GenericFunction* function_;
         mutable typestore_list_t typestore_list_;

      private:

         std::vector<expression_ptr> arg_list_;
         std::vector<branch_t>         branch_;
         mutable tmp_vs_t  expr_as_vec1_store_;
         mutable range_list_t      range_list_;
      };

      template <typename T, typename GenericFunction>
      class multimode_genfunction_node : public generic_function_node<T,GenericFunction>
      {
      public:

         typedef generic_function_node<T,GenericFunction> gen_function_t;
         typedef typename gen_function_t::range_t         range_t;

         multimode_genfunction_node(GenericFunction* func,
                                    const std::size_t& param_seq_index,
                                    const std::vector<typename gen_function_t::expression_ptr>& arg_list)
         : gen_function_t(arg_list,func)
         , param_seq_index_(param_seq_index)
         {}

         inline T value() const exprtk_override
         {
            if (gen_function_t::function_)
            {
               if (gen_function_t::populate_value_list())
               {
                  typedef typename GenericFunction::parameter_list_t parameter_list_t;

                  return (*gen_function_t::function_)
                            (
                              param_seq_index_,
                              parameter_list_t(gen_function_t::typestore_list_)
                            );
               }
            }

            return std::numeric_limits<T>::quiet_NaN();
         }

         inline typename expression_node<T>::node_type type() const exprtk_override exprtk_final
         {
            return expression_node<T>::e_genfunction;
         }

      private:

         std::size_t param_seq_index_;
      };

      class return_exception
      {};

      template <typename T>
      class null_igenfunc
      {
      public:

         virtual ~null_igenfunc() {}

         typedef type_store<T> generic_type;
         typedef typename generic_type::parameter_list parameter_list_t;

         inline virtual T operator() (parameter_list_t)
         {
            return std::numeric_limits<T>::quiet_NaN();
         }
      };

      #define exprtk_define_unary_op(OpName)                    \
      template <typename T>                                     \
      struct OpName##_op                                        \
      {                                                         \
         typedef typename functor_t<T>::Type Type;              \
         typedef typename expression_node<T>::node_type node_t; \
                                                                \
         static inline T process(Type v)                        \
         {                                                      \
            return numeric:: OpName (v);                        \
         }                                                      \
                                                                \
         static inline node_t type()                            \
         {                                                      \
            return expression_node<T>::e_##OpName;              \
         }                                                      \
                                                                \
         static inline details::operator_type operation()       \
         {                                                      \
            return details::e_##OpName;                         \
         }                                                      \
      };                                                        \

      exprtk_define_unary_op(abs  )
      exprtk_define_unary_op(acos )
      exprtk_define_unary_op(acosh)
      exprtk_define_unary_op(asin )
      exprtk_define_unary_op(asinh)
      exprtk_define_unary_op(atan )
      exprtk_define_unary_op(atanh)
      exprtk_define_unary_op(ceil )
      exprtk_define_unary_op(cos  )
      exprtk_define_unary_op(cosh )
      exprtk_define_unary_op(cot  )
      exprtk_define_unary_op(csc  )
      exprtk_define_unary_op(d2g  )
      exprtk_define_unary_op(d2r  )
      exprtk_define_unary_op(erf  )
      exprtk_define_unary_op(erfc )
      exprtk_define_unary_op(exp  )
      exprtk_define_unary_op(expm1)
      exprtk_define_unary_op(floor)
      exprtk_define_unary_op(frac )
      exprtk_define_unary_op(g2d  )
      exprtk_define_unary_op(log  )
      exprtk_define_unary_op(log10)
      exprtk_define_unary_op(log2 )
      exprtk_define_unary_op(log1p)
      exprtk_define_unary_op(ncdf )
      exprtk_define_unary_op(neg  )
      exprtk_define_unary_op(notl )
      exprtk_define_unary_op(pos  )
      exprtk_define_unary_op(r2d  )
      exprtk_define_unary_op(round)
      exprtk_define_unary_op(sec  )
      exprtk_define_unary_op(sgn  )
      exprtk_define_unary_op(sin  )
      exprtk_define_unary_op(sinc )
      exprtk_define_unary_op(sinh )
      exprtk_define_unary_op(sqrt )
      exprtk_define_unary_op(tan  )
      exprtk_define_unary_op(tanh )
      exprtk_define_unary_op(trunc)
      #undef exprtk_define_unary_op

      template <typename T>
      struct opr_base
      {
         typedef typename details::functor_t<T>::Type    Type;
         typedef typename details::functor_t<T>::RefType RefType;
         typedef typename details::functor_t<T> functor_t;
         typedef typename functor_t::qfunc_t    quaternary_functor_t;
         typedef typename functor_t::tfunc_t    trinary_functor_t;
         typedef typename functor_t::bfunc_t    binary_functor_t;
         typedef typename functor_t::ufunc_t    unary_functor_t;
      };

      template <typename T>
      struct add_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type    Type;
         typedef typename opr_base<T>::RefType RefType;

         static inline T process(Type t1, Type t2) { return t1 + t2; }
         static inline T process(Type t1, Type t2, Type t3) { return t1 + t2 + t3; }
         static inline void assign(RefType t1, Type t2) { t1 += t2; }
         static inline typename expression_node<T>::node_type type() { return expression_node<T>::e_add; }
         static inline details::operator_type operation() { return details::e_add; }
      };

      template <typename T>
      struct mul_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type    Type;
         typedef typename opr_base<T>::RefType RefType;

         static inline T process(Type t1, Type t2) { return t1 * t2; }
         static inline T process(Type t1, Type t2, Type t3) { return t1 * t2 * t3; }
         static inline void assign(RefType t1, Type t2) { t1 *= t2; }
         static inline typename expression_node<T>::node_type type() { return expression_node<T>::e_mul; }
         static inline details::operator_type operation() { return details::e_mul; }
      };

      template <typename T>
      struct sub_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type    Type;
         typedef typename opr_base<T>::RefType RefType;

         static inline T process(Type t1, Type t2) { return t1 - t2; }
         static inline T process(Type t1, Type t2, Type t3) { return t1 - t2 - t3; }
         static inline void assign(RefType t1, Type t2) { t1 -= t2; }
         static inline typename expression_node<T>::node_type type() { return expression_node<T>::e_sub; }
         static inline details::operator_type operation() { return details::e_sub; }
      };

      template <typename T>
      struct div_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type    Type;
         typedef typename opr_base<T>::RefType RefType;

         static inline T process(Type t1, Type t2) { return t1 / t2; }
         static inline T process(Type t1, Type t2, Type t3) { return t1 / t2 / t3; }
         static inline void assign(RefType t1, Type t2) { t1 /= t2; }
         static inline typename expression_node<T>::node_type type() { return expression_node<T>::e_div; }
         static inline details::operator_type operation() { return details::e_div; }
      };

      template <typename T>
      struct mod_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type    Type;
         typedef typename opr_base<T>::RefType RefType;

         static inline T process(Type t1, Type t2) { return numeric::modulus<T>(t1,t2); }
         static inline void assign(RefType t1, Type t2) { t1 = numeric::modulus<T>(t1,t2); }
         static inline typename expression_node<T>::node_type type() { return expression_node<T>::e_mod; }
         static inline details::operator_type operation() { return details::e_mod; }
      };

      template <typename T>
      struct pow_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type    Type;
         typedef typename opr_base<T>::RefType RefType;

         static inline T process(Type t1, Type t2) { return numeric::pow<T>(t1,t2); }
         static inline void assign(RefType t1, Type t2) { t1 = numeric::pow<T>(t1,t2); }
         static inline typename expression_node<T>::node_type type() { return expression_node<T>::e_pow; }
         static inline details::operator_type operation() { return details::e_pow; }
      };

      template <typename T>
      struct lt_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type Type;

         static inline T process(Type t1, Type t2) { return ((t1 < t2) ? T(1) : T(0)); }
         static inline T process(const std::string& t1, const std::string& t2) { return ((t1 < t2) ? T(1) : T(0)); }
         static inline typename expression_node<T>::node_type type() { return expression_node<T>::e_lt; }
         static inline details::operator_type operation() { return details::e_lt; }
      };

      template <typename T>
      struct lte_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type Type;

         static inline T process(Type t1, Type t2) { return ((t1 <= t2) ? T(1) : T(0)); }
         static inline T process(const std::string& t1, const std::string& t2) { return ((t1 <= t2) ? T(1) : T(0)); }
         static inline typename expression_node<T>::node_type type() { return expression_node<T>::e_lte; }
         static inline details::operator_type operation() { return details::e_lte; }
      };

      template <typename T>
      struct gt_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type Type;

         static inline T process(Type t1, Type t2) { return ((t1 > t2) ? T(1) : T(0)); }
         static inline T process(const std::string& t1, const std::string& t2) { return ((t1 > t2) ? T(1) : T(0)); }
         static inline typename expression_node<T>::node_type type() { return expression_node<T>::e_gt; }
         static inline details::operator_type operation() { return details::e_gt; }
      };

      template <typename T>
      struct gte_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type Type;

         static inline T process(Type t1, Type t2) { return ((t1 >= t2) ? T(1) : T(0)); }
         static inline T process(const std::string& t1, const std::string& t2) { return ((t1 >= t2) ? T(1) : T(0)); }
         static inline typename expression_node<T>::node_type type() { return expression_node<T>::e_gte; }
         static inline details::operator_type operation() { return details::e_gte; }
      };

      template <typename T>
      struct eq_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type Type;
         static inline T process(Type t1, Type t2) { return (std::equal_to<T>()(t1,t2) ? T(1) : T(0)); }
         static inline T process(const std::string& t1, const std::string& t2) { return ((t1 == t2) ? T(1) : T(0)); }
         static inline typename expression_node<T>::node_type type() { return expression_node<T>::e_eq; }
         static inline details::operator_type operation() { return details::e_eq; }
      };

      template <typename T>
      struct equal_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type Type;

         static inline T process(Type t1, Type t2) { return numeric::equal(t1,t2); }
         static inline T process(const std::string& t1, const std::string& t2) { return ((t1 == t2) ? T(1) : T(0)); }
         static inline typename expression_node<T>::node_type type() { return expression_node<T>::e_eq; }
         static inline details::operator_type operation() { return details::e_equal; }
      };

      template <typename T>
      struct ne_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type Type;

         static inline T process(Type t1, Type t2) { return (std::not_equal_to<T>()(t1,t2) ? T(1) : T(0)); }
         static inline T process(const std::string& t1, const std::string& t2) { return ((t1 != t2) ? T(1) : T(0)); }
         static inline typename expression_node<T>::node_type type() { return expression_node<T>::e_ne; }
         static inline details::operator_type operation() { return details::e_ne; }
      };

      template <typename T>
      struct and_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type Type;

         static inline T process(Type t1, Type t2) { return (details::is_true(t1) && details::is_true(t2)) ? T(1) : T(0); }
         static inline typename expression_node<T>::node_type type() { return expression_node<T>::e_and; }
         static inline details::operator_type operation() { return details::e_and; }
      };

      template <typename T>
      struct nand_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type Type;

         static inline T process(Type t1, Type t2) { return (details::is_true(t1) && details::is_true(t2)) ? T(0) : T(1); }
         static inline typename expression_node<T>::node_type type() { return expression_node<T>::e_nand; }
         static inline details::operator_type operation() { return details::e_nand; }
      };

      template <typename T>
      struct or_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type Type;

         static inline T process(Type t1, Type t2) { return (details::is_true(t1) || details::is_true(t2)) ? T(1) : T(0); }
         static inline typename expression_node<T>::node_type type() { return expression_node<T>::e_or; }
         static inline details::operator_type operation() { return details::e_or; }
      };

      template <typename T>
      struct nor_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type Type;

         static inline T process(Type t1, Type t2) { return (details::is_true(t1) || details::is_true(t2)) ? T(0) : T(1); }
         static inline typename expression_node<T>::node_type type() { return expression_node<T>::e_nor; }
         static inline details::operator_type operation() { return details::e_nor; }
      };

      template <typename T>
      struct xor_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type Type;

         static inline T process(Type t1, Type t2) { return numeric::xor_opr<T>(t1,t2); }
         static inline typename expression_node<T>::node_type type() { return expression_node<T>::e_nor; }
         static inline details::operator_type operation() { return details::e_xor; }
      };

      template <typename T>
      struct xnor_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type Type;

         static inline T process(Type t1, Type t2) { return numeric::xnor_opr<T>(t1,t2); }
         static inline typename expression_node<T>::node_type type() { return expression_node<T>::e_nor; }
         static inline details::operator_type operation() { return details::e_xnor; }
      };

      template <typename T>
      struct in_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type Type;

         static inline T process(const T&, const T&) { return std::numeric_limits<T>::quiet_NaN(); }
         static inline T process(const std::string& t1, const std::string& t2) { return ((std::string::npos != t2.find(t1)) ? T(1) : T(0)); }
         static inline typename expression_node<T>::node_type type() { return expression_node<T>::e_in; }
         static inline details::operator_type operation() { return details::e_in; }
      };

      template <typename T>
      struct like_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type Type;

         static inline T process(const T&, const T&) { return std::numeric_limits<T>::quiet_NaN(); }
         static inline T process(const std::string& t1, const std::string& t2) { return (details::wc_match(t2,t1) ? T(1) : T(0)); }
         static inline typename expression_node<T>::node_type type() { return expression_node<T>::e_like; }
         static inline details::operator_type operation() { return details::e_like; }
      };

      template <typename T>
      struct ilike_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type Type;

         static inline T process(const T&, const T&) { return std::numeric_limits<T>::quiet_NaN(); }
         static inline T process(const std::string& t1, const std::string& t2) { return (details::wc_imatch(t2,t1) ? T(1) : T(0)); }
         static inline typename expression_node<T>::node_type type() { return expression_node<T>::e_ilike; }
         static inline details::operator_type operation() { return details::e_ilike; }
      };

      template <typename T>
      struct inrange_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type Type;

         static inline T process(const T& t0, const T& t1, const T& t2) { return ((t0 <= t1) && (t1 <= t2)) ? T(1) : T(0); }
         static inline T process(const std::string& t0, const std::string& t1, const std::string& t2)
         {
            return ((t0 <= t1) && (t1 <= t2)) ? T(1) : T(0);
         }
         static inline typename expression_node<T>::node_type type() { return expression_node<T>::e_inranges; }
         static inline details::operator_type operation() { return details::e_inrange; }
      };

      template <typename T>
      inline T value(details::expression_node<T>* n)
      {
         return n->value();
      }

      template <typename T>
      inline T value(std::pair<details::expression_node<T>*,bool> n)
      {
         return n.first->value();
      }

      template <typename T>
      inline T value(const T* t)
      {
         return (*t);
      }

      template <typename T>
      inline T value(const T& t)
      {
         return t;
      }

      template <typename T>
      struct vararg_add_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type Type;

         template <typename Type,
                   typename Allocator,
                   template <typename, typename> class Sequence>
         static inline T process(const Sequence<Type,Allocator>& arg_list)
         {
            switch (arg_list.size())
            {
               case 0  : return T(0);
               case 1  : return process_1(arg_list);
               case 2  : return process_2(arg_list);
               case 3  : return process_3(arg_list);
               case 4  : return process_4(arg_list);
               case 5  : return process_5(arg_list);
               default :
                         {
                            T result = T(0);

                            for (std::size_t i = 0; i < arg_list.size(); ++i)
                            {
                              result += value(arg_list[i]);
                            }

                            return result;
                         }
            }
         }

         template <typename Sequence>
         static inline T process_1(const Sequence& arg_list)
         {
            return value(arg_list[0]);
         }

         template <typename Sequence>
         static inline T process_2(const Sequence& arg_list)
         {
            return value(arg_list[0]) + value(arg_list[1]);
         }

         template <typename Sequence>
         static inline T process_3(const Sequence& arg_list)
         {
            return value(arg_list[0]) + value(arg_list[1]) +
                   value(arg_list[2]) ;
         }

         template <typename Sequence>
         static inline T process_4(const Sequence& arg_list)
         {
            return value(arg_list[0]) + value(arg_list[1]) +
                   value(arg_list[2]) + value(arg_list[3]) ;
         }

         template <typename Sequence>
         static inline T process_5(const Sequence& arg_list)
         {
            return value(arg_list[0]) + value(arg_list[1]) +
                   value(arg_list[2]) + value(arg_list[3]) +
                   value(arg_list[4]) ;
         }
      };

      template <typename T>
      struct vararg_mul_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type Type;

         template <typename Type,
                   typename Allocator,
                   template <typename, typename> class Sequence>
         static inline T process(const Sequence<Type,Allocator>& arg_list)
         {
            switch (arg_list.size())
            {
               case 0  : return T(0);
               case 1  : return process_1(arg_list);
               case 2  : return process_2(arg_list);
               case 3  : return process_3(arg_list);
               case 4  : return process_4(arg_list);
               case 5  : return process_5(arg_list);
               default :
                         {
                            T result = T(value(arg_list[0]));

                            for (std::size_t i = 1; i < arg_list.size(); ++i)
                            {
                               result *= value(arg_list[i]);
                            }

                            return result;
                         }
            }
         }

         template <typename Sequence>
         static inline T process_1(const Sequence& arg_list)
         {
            return value(arg_list[0]);
         }

         template <typename Sequence>
         static inline T process_2(const Sequence& arg_list)
         {
            return value(arg_list[0]) * value(arg_list[1]);
         }

         template <typename Sequence>
         static inline T process_3(const Sequence& arg_list)
         {
            return value(arg_list[0]) * value(arg_list[1]) *
                   value(arg_list[2]) ;
         }

         template <typename Sequence>
         static inline T process_4(const Sequence& arg_list)
         {
            return value(arg_list[0]) * value(arg_list[1]) *
                   value(arg_list[2]) * value(arg_list[3]) ;
         }

         template <typename Sequence>
         static inline T process_5(const Sequence& arg_list)
         {
            return value(arg_list[0]) * value(arg_list[1]) *
                   value(arg_list[2]) * value(arg_list[3]) *
                   value(arg_list[4]) ;
         }
      };

      template <typename T>
      struct vararg_avg_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type Type;

         template <typename Type,
                   typename Allocator,
                   template <typename, typename> class Sequence>
         static inline T process(const Sequence<Type,Allocator>& arg_list)
         {
            switch (arg_list.size())
            {
               case 0  : return T(0);
               case 1  : return process_1(arg_list);
               case 2  : return process_2(arg_list);
               case 3  : return process_3(arg_list);
               case 4  : return process_4(arg_list);
               case 5  : return process_5(arg_list);
               default : return vararg_add_op<T>::process(arg_list) / arg_list.size();
            }
         }

         template <typename Sequence>
         static inline T process_1(const Sequence& arg_list)
         {
            return value(arg_list[0]);
         }

         template <typename Sequence>
         static inline T process_2(const Sequence& arg_list)
         {
            return (value(arg_list[0]) + value(arg_list[1])) / T(2);
         }

         template <typename Sequence>
         static inline T process_3(const Sequence& arg_list)
         {
            return (value(arg_list[0]) + value(arg_list[1]) + value(arg_list[2])) / T(3);
         }

         template <typename Sequence>
         static inline T process_4(const Sequence& arg_list)
         {
            return (value(arg_list[0]) + value(arg_list[1]) +
                    value(arg_list[2]) + value(arg_list[3])) / T(4);
         }

         template <typename Sequence>
         static inline T process_5(const Sequence& arg_list)
         {
            return (value(arg_list[0]) + value(arg_list[1]) +
                    value(arg_list[2]) + value(arg_list[3]) +
                    value(arg_list[4])) / T(5);
         }
      };

      template <typename T>
      struct vararg_min_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type Type;

         template <typename Type,
                   typename Allocator,
                   template <typename, typename> class Sequence>
         static inline T process(const Sequence<Type,Allocator>& arg_list)
         {
            switch (arg_list.size())
            {
               case 0  : return T(0);
               case 1  : return process_1(arg_list);
               case 2  : return process_2(arg_list);
               case 3  : return process_3(arg_list);
               case 4  : return process_4(arg_list);
               case 5  : return process_5(arg_list);
               default :
                         {
                            T result = T(value(arg_list[0]));

                            for (std::size_t i = 1; i < arg_list.size(); ++i)
                            {
                               const T v = value(arg_list[i]);

                               if (v < result)
                                  result = v;
                            }

                            return result;
                         }
            }
         }

         template <typename Sequence>
         static inline T process_1(const Sequence& arg_list)
         {
            return value(arg_list[0]);
         }

         template <typename Sequence>
         static inline T process_2(const Sequence& arg_list)
         {
            return std::min<T>(value(arg_list[0]),value(arg_list[1]));
         }

         template <typename Sequence>
         static inline T process_3(const Sequence& arg_list)
         {
            return std::min<T>(std::min<T>(value(arg_list[0]),value(arg_list[1])),value(arg_list[2]));
         }

         template <typename Sequence>
         static inline T process_4(const Sequence& arg_list)
         {
            return std::min<T>(
                        std::min<T>(value(arg_list[0]), value(arg_list[1])),
                        std::min<T>(value(arg_list[2]), value(arg_list[3])));
         }

         template <typename Sequence>
         static inline T process_5(const Sequence& arg_list)
         {
            return std::min<T>(
                   std::min<T>(std::min<T>(value(arg_list[0]), value(arg_list[1])),
                               std::min<T>(value(arg_list[2]), value(arg_list[3]))),
                               value(arg_list[4]));
         }
      };

      template <typename T>
      struct vararg_max_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type Type;

         template <typename Type,
                   typename Allocator,
                   template <typename, typename> class Sequence>
         static inline T process(const Sequence<Type,Allocator>& arg_list)
         {
            switch (arg_list.size())
            {
               case 0  : return T(0);
               case 1  : return process_1(arg_list);
               case 2  : return process_2(arg_list);
               case 3  : return process_3(arg_list);
               case 4  : return process_4(arg_list);
               case 5  : return process_5(arg_list);
               default :
                         {
                            T result = T(value(arg_list[0]));

                            for (std::size_t i = 1; i < arg_list.size(); ++i)
                            {
                               const T v = value(arg_list[i]);

                               if (v > result)
                                  result = v;
                            }

                            return result;
                         }
            }
         }

         template <typename Sequence>
         static inline T process_1(const Sequence& arg_list)
         {
            return value(arg_list[0]);
         }

         template <typename Sequence>
         static inline T process_2(const Sequence& arg_list)
         {
            return std::max<T>(value(arg_list[0]),value(arg_list[1]));
         }

         template <typename Sequence>
         static inline T process_3(const Sequence& arg_list)
         {
            return std::max<T>(std::max<T>(value(arg_list[0]),value(arg_list[1])),value(arg_list[2]));
         }

         template <typename Sequence>
         static inline T process_4(const Sequence& arg_list)
         {
            return std::max<T>(
                        std::max<T>(value(arg_list[0]), value(arg_list[1])),
                        std::max<T>(value(arg_list[2]), value(arg_list[3])));
         }

         template <typename Sequence>
         static inline T process_5(const Sequence& arg_list)
         {
            return std::max<T>(
                   std::max<T>(std::max<T>(value(arg_list[0]), value(arg_list[1])),
                               std::max<T>(value(arg_list[2]), value(arg_list[3]))),
                               value(arg_list[4]));
         }
      };

      template <typename T>
      struct vararg_mand_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type Type;

         template <typename Type,
                   typename Allocator,
                   template <typename, typename> class Sequence>
         static inline T process(const Sequence<Type,Allocator>& arg_list)
         {
            switch (arg_list.size())
            {
               case 1  : return process_1(arg_list);
               case 2  : return process_2(arg_list);
               case 3  : return process_3(arg_list);
               case 4  : return process_4(arg_list);
               case 5  : return process_5(arg_list);
               default :
                         {
                            for (std::size_t i = 0; i < arg_list.size(); ++i)
                            {
                               if (std::equal_to<T>()(T(0), value(arg_list[i])))
                                  return T(0);
                            }

                            return T(1);
                         }
            }
         }

         template <typename Sequence>
         static inline T process_1(const Sequence& arg_list)
         {
            return std::not_equal_to<T>()
                      (T(0), value(arg_list[0])) ? T(1) : T(0);
         }

         template <typename Sequence>
         static inline T process_2(const Sequence& arg_list)
         {
            return (
                     std::not_equal_to<T>()(T(0), value(arg_list[0])) &&
                     std::not_equal_to<T>()(T(0), value(arg_list[1]))
                   ) ? T(1) : T(0);
         }

         template <typename Sequence>
         static inline T process_3(const Sequence& arg_list)
         {
            return (
                     std::not_equal_to<T>()(T(0), value(arg_list[0])) &&
                     std::not_equal_to<T>()(T(0), value(arg_list[1])) &&
                     std::not_equal_to<T>()(T(0), value(arg_list[2]))
                   ) ? T(1) : T(0);
         }

         template <typename Sequence>
         static inline T process_4(const Sequence& arg_list)
         {
            return (
                     std::not_equal_to<T>()(T(0), value(arg_list[0])) &&
                     std::not_equal_to<T>()(T(0), value(arg_list[1])) &&
                     std::not_equal_to<T>()(T(0), value(arg_list[2])) &&
                     std::not_equal_to<T>()(T(0), value(arg_list[3]))
                   ) ? T(1) : T(0);
         }

         template <typename Sequence>
         static inline T process_5(const Sequence& arg_list)
         {
            return (
                     std::not_equal_to<T>()(T(0), value(arg_list[0])) &&
                     std::not_equal_to<T>()(T(0), value(arg_list[1])) &&
                     std::not_equal_to<T>()(T(0), value(arg_list[2])) &&
                     std::not_equal_to<T>()(T(0), value(arg_list[3])) &&
                     std::not_equal_to<T>()(T(0), value(arg_list[4]))
                   ) ? T(1) : T(0);
         }
      };

      template <typename T>
      struct vararg_mor_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type Type;

         template <typename Type,
                   typename Allocator,
                   template <typename, typename> class Sequence>
         static inline T process(const Sequence<Type,Allocator>& arg_list)
         {
            switch (arg_list.size())
            {
               case 1  : return process_1(arg_list);
               case 2  : return process_2(arg_list);
               case 3  : return process_3(arg_list);
               case 4  : return process_4(arg_list);
               case 5  : return process_5(arg_list);
               default :
                         {
                            for (std::size_t i = 0; i < arg_list.size(); ++i)
                            {
                               if (std::not_equal_to<T>()(T(0), value(arg_list[i])))
                                  return T(1);
                            }

                            return T(0);
                         }
            }
         }

         template <typename Sequence>
         static inline T process_1(const Sequence& arg_list)
         {
            return std::not_equal_to<T>()
                      (T(0), value(arg_list[0])) ? T(1) : T(0);
         }

         template <typename Sequence>
         static inline T process_2(const Sequence& arg_list)
         {
            return (
                     std::not_equal_to<T>()(T(0), value(arg_list[0])) ||
                     std::not_equal_to<T>()(T(0), value(arg_list[1]))
                   ) ? T(1) : T(0);
         }

         template <typename Sequence>
         static inline T process_3(const Sequence& arg_list)
         {
            return (
                     std::not_equal_to<T>()(T(0), value(arg_list[0])) ||
                     std::not_equal_to<T>()(T(0), value(arg_list[1])) ||
                     std::not_equal_to<T>()(T(0), value(arg_list[2]))
                   ) ? T(1) : T(0);
         }

         template <typename Sequence>
         static inline T process_4(const Sequence& arg_list)
         {
            return (
                     std::not_equal_to<T>()(T(0), value(arg_list[0])) ||
                     std::not_equal_to<T>()(T(0), value(arg_list[1])) ||
                     std::not_equal_to<T>()(T(0), value(arg_list[2])) ||
                     std::not_equal_to<T>()(T(0), value(arg_list[3]))
                   ) ? T(1) : T(0);
         }

         template <typename Sequence>
         static inline T process_5(const Sequence& arg_list)
         {
            return (
                     std::not_equal_to<T>()(T(0), value(arg_list[0])) ||
                     std::not_equal_to<T>()(T(0), value(arg_list[1])) ||
                     std::not_equal_to<T>()(T(0), value(arg_list[2])) ||
                     std::not_equal_to<T>()(T(0), value(arg_list[3])) ||
                     std::not_equal_to<T>()(T(0), value(arg_list[4]))
                   ) ? T(1) : T(0);
         }
      };

      template <typename T>
      struct vararg_multi_op : public opr_base<T>
      {
         typedef typename opr_base<T>::Type Type;

         template <typename Type,
                   typename Allocator,
                   template <typename, typename> class Sequence>
         static inline T process(const Sequence<Type,Allocator>& arg_list)
         {
            switch (arg_list.size())
            {
               case 0  : return std::numeric_limits<T>::quiet_NaN();
               case 1  : return process_1(arg_list);
               case 2  : return process_2(arg_list);
               case 3  : return process_3(arg_list);
               case 4  : return process_4(arg_list);
               case 5  : return process_5(arg_list);
               case 6  : return process_6(arg_list);
               case 7  : return process_7(arg_list);
               case 8  : return process_8(arg_list);
               default :
                         {
                            for (std::size_t i = 0; i < (arg_list.size() - 1); ++i)
                            {
                               value(arg_list[i]);
                            }

                            return value(arg_list.back());
                         }
            }
         }

         template <typename Sequence>
         static inline T process_1(const Sequence& arg_list)
         {
            return value(arg_list[0]);
         }

         template <typename Sequence>
         static inline T process_2(const Sequence& arg_list)
         {
                   value(arg_list[0]);
            return value(arg_list[1]);
         }

         template <typename Sequence>
         static inline T process_3(const Sequence& arg_list)
         {
                   value(arg_list[0]);
                   value(arg_list[1]);
            return value(arg_list[2]);
         }

         template <typename Sequence>
         static inline T process_4(const Sequence& arg_list)
         {
                   value(arg_list[0]);
                   value(arg_list[1]);
                   value(arg_list[2]);
            return value(arg_list[3]);
         }

         template <typename Sequence>
         static inline T process_5(const Sequence& arg_list)
         {
                   value(arg_list[0]);
                   value(arg_list[1]);
                   value(arg_list[2]);
                   value(arg_list[3]);
            return value(arg_list[4]);
         }

         template <typename Sequence>
         static inline T process_6(const Sequence& arg_list)
         {
                   value(arg_list[0]);
                   value(arg_list[1]);
                   value(arg_list[2]);
                   value(arg_list[3]);
                   value(arg_list[4]);
            return value(arg_list[5]);
         }

         template <typename Sequence>
         static inline T process_7(const Sequence& arg_list)
         {
                   value(arg_list[0]);
                   value(arg_list[1]);
                   value(arg_list[2]);
                   value(arg_list[3]);
                   value(arg_list[4]);
                   value(arg_list[5]);
            return value(arg_list[6]);
         }

         template <typename Sequence>
         static inline T process_8(const Sequence& arg_list)
         {
                   value(arg_list[0]);
                   value(arg_list[1]);
                   value(arg_list[2]);
                   value(arg_list[3]);
                   value(arg_list[4]);
                   value(arg_list[5]);
                   value(arg_list[6]);
            return value(arg_list[7]);
         }
      };

      template <typename T>
      struct vec_add_op
      {
         typedef vector_interface<T>* ivector_ptr;

         static inline T process(const ivector_ptr v)
         {
            const T* vec = v->vec()->vds().data();
            const std::size_t vec_size = v->vec()->vds().size();

            loop_unroll::details lud(vec_size);

            if (vec_size <= static_cast<std::size_t>(lud.batch_size))
            {
               T result = T(0);
               int i    = 0;

               exprtk_disable_fallthrough_begin
               switch (vec_size)
               {
                  #define case_stmt(N)         \
                  case N : result += vec[i++]; \

                  case_stmt( 4) case_stmt( 3)
                  case_stmt( 2) case_stmt( 1)
               }
               exprtk_disable_fallthrough_end

               #undef case_stmt

               return result;
            }

            T r[] = {
                      T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0),
                      T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0)
                    };

            const T* upper_bound = vec + lud.upper_bound;

            while (vec < upper_bound)
            {
               #define exprtk_loop(N) \
               r[N] += vec[N];        \

               exprtk_loop( 0) exprtk_loop( 1)
               exprtk_loop( 2) exprtk_loop( 3)

               vec += lud.batch_size;
            }

            int i = 0;

            exprtk_disable_fallthrough_begin
            switch (lud.remainder)
            {
               #define case_stmt(N)       \
               case N : r[0] += vec[i++]; \

               case_stmt( 3) case_stmt( 2)
               case_stmt( 1)
            }
            exprtk_disable_fallthrough_end

            #undef exprtk_loop
            #undef case_stmt

            return (r[ 0] + r[ 1] + r[ 2] + r[ 3])
                   #ifndef exprtk_disable_superscalar_unroll
                 + (r[ 4] + r[ 5] + r[ 6] + r[ 7])
                 + (r[ 8] + r[ 9] + r[10] + r[11])
                 + (r[12] + r[13] + r[14] + r[15])
                   #endif
                   ;
         }
      };

      template <typename T>
      struct vec_mul_op
      {
         typedef vector_interface<T>* ivector_ptr;

         static inline T process(const ivector_ptr v)
         {
            const T* vec = v->vec()->vds().data();
            const std::size_t vec_size = v->vec()->vds().size();

            loop_unroll::details lud(vec_size);

            if (vec_size <= static_cast<std::size_t>(lud.batch_size))
            {
               T result = T(1);
               int i    = 0;

               exprtk_disable_fallthrough_begin
               switch (vec_size)
               {
                  #define case_stmt(N)         \
                  case N : result *= vec[i++]; \

                  case_stmt( 4) case_stmt( 3)
                  case_stmt( 2) case_stmt( 1)
               }
               exprtk_disable_fallthrough_end

               #undef case_stmt

               return result;
            }

            T r[] = {
                      T(1), T(1), T(1), T(1), T(1), T(1), T(1), T(1),
                      T(1), T(1), T(1), T(1), T(1), T(1), T(1), T(1)
                    };

            const T* upper_bound = vec + lud.upper_bound;

            while (vec < upper_bound)
            {
               #define exprtk_loop(N) \
               r[N] *= vec[N];        \

               exprtk_loop( 0) exprtk_loop( 1)
               exprtk_loop( 2) exprtk_loop( 3)

               vec += lud.batch_size;
            }

            int i = 0;

            exprtk_disable_fallthrough_begin
            switch (lud.remainder)
            {
               #define case_stmt(N)       \
               case N : r[0] *= vec[i++]; \

               case_stmt( 3) case_stmt( 2)
               case_stmt( 1)
            }
            exprtk_disable_fallthrough_end

            #undef exprtk_loop
            #undef case_stmt

            return (r[ 0] * r[ 1] * r[ 2] * r[ 3])
                   #ifndef exprtk_disable_superscalar_unroll
                 + (r[ 4] * r[ 5] * r[ 6] * r[ 7])
                 + (r[ 8] * r[ 9] * r[10] * r[11])
                 + (r[12] * r[13] * r[14] * r[15])
                   #endif
                   ;
         }
      };

      template <typename T>
      struct vec_avg_op
      {
         typedef vector_interface<T>* ivector_ptr;

         static inline T process(const ivector_ptr v)
         {
            const std::size_t vec_size = v->vec()->vds().size();

            return vec_add_op<T>::process(v) / vec_size;
         }
      };

      template <typename T>
      struct vec_min_op
      {
         typedef vector_interface<T>* ivector_ptr;

         static inline T process(const ivector_ptr v)
         {
            const T* vec = v->vec()->vds().data();
            const std::size_t vec_size = v->vec()->vds().size();

            T result = vec[0];

            for (std::size_t i = 1; i < vec_size; ++i)
            {
               const T v_i = vec[i];

               if (v_i < result)
                  result = v_i;
            }

            return result;
         }
      };

      template <typename T>
      struct vec_max_op
      {
         typedef vector_interface<T>* ivector_ptr;

         static inline T process(const ivector_ptr v)
         {
            const T* vec = v->vec()->vds().data();
            const std::size_t vec_size = v->vec()->vds().size();

            T result = vec[0];

            for (std::size_t i = 1; i < vec_size; ++i)
            {
               const T v_i = vec[i];

               if (v_i > result)
                  result = v_i;
            }

            return result;
         }
      };

      template <typename T>
      class vov_base_node : public expression_node<T>
      {
      public:

         virtual ~vov_base_node() {}

         inline virtual operator_type operation() const
         {
            return details::e_default;
         }

         virtual const T& v0() const = 0;

         virtual const T& v1() const = 0;
      };

      template <typename T>
      class cov_base_node : public expression_node<T>
      {
      public:

         virtual ~cov_base_node() {}

         inline virtual operator_type operation() const
         {
            return details::e_default;
         }

         virtual const T c() const = 0;

         virtual const T& v() const = 0;
      };

      template <typename T>
      class voc_base_node : public expression_node<T>
      {
      public:

         virtual ~voc_base_node() {}

         inline virtual operator_type operation() const
         {
            return details::e_default;
         }

         virtual const T c() const = 0;

         virtual const T& v() const = 0;
      };

      template <typename T>
      class vob_base_node : public expression_node<T>
      {
      public:

         virtual ~vob_base_node() {}

         virtual const T& v() const = 0;
      };

      template <typename T>
      class bov_base_node : public expression_node<T>
      {
      public:

         virtual ~bov_base_node() {}

         virtual const T& v() const = 0;
      };

      template <typename T>
      class cob_base_node : public expression_node<T>
      {
      public:

         virtual ~cob_base_node() {}

         inline virtual operator_type operation() const
         {
            return details::e_default;
         }

         virtual const T c() const = 0;

         virtual void set_c(const T) = 0;

         virtual expression_node<T>* move_branch(const std::size_t& index) = 0;
      };

      template <typename T>
      class boc_base_node : public expression_node<T>
      {
      public:

         virtual ~boc_base_node() {}

         inline virtual operator_type operation() const
         {
            return details::e_default;
         }

         virtual const T c() const = 0;

         virtual void set_c(const T) = 0;

         virtual expression_node<T>* move_branch(const std::size_t& index) = 0;
      };

      template <typename T>
      class uv_base_node : public expression_node<T>
      {
      public:

         virtual ~uv_base_node() {}

         inline virtual operator_type operation() const
         {
            return details::e_default;
         }

         virtual const T& v() const = 0;
      };

      template <typename T>
      class sos_base_node : public expression_node<T>
      {
      public:

         virtual ~sos_base_node() {}

         inline virtual operator_type operation() const
         {
            return details::e_default;
         }
      };

      template <typename T>
      class sosos_base_node : public expression_node<T>
      {
      public:

         virtual ~sosos_base_node() {}

         inline virtual operator_type operation() const
         {
            return details::e_default;
         }
      };

      template <typename T>
      class T0oT1oT2_base_node : public expression_node<T>
      {
      public:

         virtual ~T0oT1oT2_base_node() {}

         virtual std::string type_id() const = 0;
      };

      template <typename T>
      class T0oT1oT2oT3_base_node : public expression_node<T>
      {
      public:

         virtual ~T0oT1oT2oT3_base_node() {}

         virtual std::string type_id() const = 0;
      };

      template <typename T, typename Operation>
      class unary_variable_node exprtk_final : public uv_base_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef Operation operation_t;

         explicit unary_variable_node(const T& var)
         : v_(var)
         {}

         inline T value() const exprtk_override
         {
            return Operation::process(v_);
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return Operation::type();
         }

         inline operator_type operation() const exprtk_override
         {
            return Operation::operation();
         }

         inline const T& v() const exprtk_override
         {
            return v_;
         }

      private:

         unary_variable_node(const unary_variable_node<T,Operation>&) exprtk_delete;
         unary_variable_node<T,Operation>& operator=(const unary_variable_node<T,Operation>&) exprtk_delete;

         const T& v_;
      };

      template <typename T>
      class uvouv_node exprtk_final : public expression_node<T>
      {
      public:

         // UOpr1(v0) Op UOpr2(v1)
         typedef typename details::functor_t<T> functor_t;
         typedef typename functor_t::bfunc_t    bfunc_t;
         typedef typename functor_t::ufunc_t    ufunc_t;
         typedef expression_node<T>*            expression_ptr;

         explicit uvouv_node(const T& var0,const T& var1,
                             ufunc_t uf0, ufunc_t uf1, bfunc_t bf)
         : v0_(var0)
         , v1_(var1)
         , u0_(uf0 )
         , u1_(uf1 )
         , f_ (bf  )
         {}

         inline T value() const exprtk_override
         {
            return f_(u0_(v0_),u1_(v1_));
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_uvouv;
         }

         inline const T& v0()
         {
            return v0_;
         }

         inline const T& v1()
         {
            return v1_;
         }

         inline ufunc_t u0()
         {
            return u0_;
         }

         inline ufunc_t u1()
         {
            return u1_;
         }

         inline ufunc_t f()
         {
            return f_;
         }

      private:

         uvouv_node(const uvouv_node<T>&) exprtk_delete;
         uvouv_node<T>& operator=(const uvouv_node<T>&) exprtk_delete;

         const T& v0_;
         const T& v1_;
         const ufunc_t u0_;
         const ufunc_t u1_;
         const bfunc_t f_;
      };

      template <typename T, typename Operation>
      class unary_branch_node exprtk_final : public expression_node<T>
      {
      public:

         typedef Operation                      operation_t;
         typedef expression_node<T>*            expression_ptr;
         typedef std::pair<expression_ptr,bool> branch_t;

         explicit unary_branch_node(expression_ptr branch)
         {
            construct_branch_pair(branch_, branch);
         }

         inline T value() const exprtk_override
         {
            return Operation::process(branch_.first->value());
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return Operation::type();
         }

         inline operator_type operation()
         {
            return Operation::operation();
         }

         inline expression_node<T>* branch(const std::size_t&) const exprtk_override
         {
            return branch_.first;
         }

         inline void release()
         {
            branch_.second = false;
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(branch_, node_delete_list);
         }

         std::size_t node_depth() const exprtk_override
         {
            return expression_node<T>::ndb_t::compute_node_depth(branch_);
         }

      private:

         unary_branch_node(const unary_branch_node<T,Operation>&) exprtk_delete;
         unary_branch_node<T,Operation>& operator=(const unary_branch_node<T,Operation>&) exprtk_delete;

         branch_t branch_;
      };

      template <typename T> struct is_const                { enum {result = 0}; };
      template <typename T> struct is_const <const T>      { enum {result = 1}; };
      template <typename T> struct is_const_ref            { enum {result = 0}; };
      template <typename T> struct is_const_ref <const T&> { enum {result = 1}; };
      template <typename T> struct is_ref                  { enum {result = 0}; };
      template <typename T> struct is_ref<T&>              { enum {result = 1}; };
      template <typename T> struct is_ref<const T&>        { enum {result = 0}; };

      template <std::size_t State>
      struct param_to_str { static std::string result() { static const std::string r("v"); return r; } };

      template <>
      struct param_to_str<0> { static std::string result() { static const std::string r("c"); return r; } };

      #define exprtk_crtype(Type)                          \
      param_to_str<is_const_ref< Type >::result>::result() \

      template <typename T>
      struct T0oT1oT2process
      {
         typedef typename details::functor_t<T> functor_t;
         typedef typename functor_t::bfunc_t    bfunc_t;

         struct mode0
         {
            static inline T process(const T& t0, const T& t1, const T& t2, const bfunc_t bf0, const bfunc_t bf1)
            {
               // (T0 o0 T1) o1 T2
               return bf1(bf0(t0,t1),t2);
            }

            template <typename T0, typename T1, typename T2>
            static inline std::string id()
            {
               static const std::string result = "(" + exprtk_crtype(T0) + "o"   +
                                                       exprtk_crtype(T1) + ")o(" +
                                                       exprtk_crtype(T2) + ")"   ;
               return result;
            }
         };

         struct mode1
         {
            static inline T process(const T& t0, const T& t1, const T& t2, const bfunc_t bf0, const bfunc_t bf1)
            {
               // T0 o0 (T1 o1 T2)
               return bf0(t0,bf1(t1,t2));
            }

            template <typename T0, typename T1, typename T2>
            static inline std::string id()
            {
               static const std::string result = "(" + exprtk_crtype(T0) + ")o(" +
                                                       exprtk_crtype(T1) + "o"   +
                                                       exprtk_crtype(T2) + ")"   ;
               return result;
            }
         };
      };

      template <typename T>
      struct T0oT1oT20T3process
      {
         typedef typename details::functor_t<T> functor_t;
         typedef typename functor_t::bfunc_t    bfunc_t;

         struct mode0
         {
            static inline T process(const T& t0, const T& t1,
                                    const T& t2, const T& t3,
                                    const bfunc_t bf0, const bfunc_t bf1, const bfunc_t bf2)
            {
               // (T0 o0 T1) o1 (T2 o2 T3)
               return bf1(bf0(t0,t1),bf2(t2,t3));
            }

            template <typename T0, typename T1, typename T2, typename T3>
            static inline std::string id()
            {
               static const std::string result = "(" + exprtk_crtype(T0) + "o"  +
                                                       exprtk_crtype(T1) + ")o" +
                                                 "(" + exprtk_crtype(T2) + "o"  +
                                                       exprtk_crtype(T3) + ")"  ;
               return result;
            }
         };

         struct mode1
         {
            static inline T process(const T& t0, const T& t1,
                                    const T& t2, const T& t3,
                                    const bfunc_t bf0, const bfunc_t bf1, const bfunc_t bf2)
            {
               // (T0 o0 (T1 o1 (T2 o2 T3))
               return bf0(t0,bf1(t1,bf2(t2,t3)));
            }
            template <typename T0, typename T1, typename T2, typename T3>
            static inline std::string id()
            {
               static const std::string result = "(" + exprtk_crtype(T0) +  ")o((" +
                                                       exprtk_crtype(T1) +  ")o("  +
                                                       exprtk_crtype(T2) +  "o"    +
                                                       exprtk_crtype(T3) +  "))"   ;
               return result;
            }
         };

         struct mode2
         {
            static inline T process(const T& t0, const T& t1,
                                    const T& t2, const T& t3,
                                    const bfunc_t bf0, const bfunc_t bf1, const bfunc_t bf2)
            {
               // (T0 o0 ((T1 o1 T2) o2 T3)
               return bf0(t0,bf2(bf1(t1,t2),t3));
            }

            template <typename T0, typename T1, typename T2, typename T3>
            static inline std::string id()
            {
               static const std::string result = "(" + exprtk_crtype(T0) + ")o((" +
                                                       exprtk_crtype(T1) + "o"    +
                                                       exprtk_crtype(T2) + ")o("  +
                                                       exprtk_crtype(T3) + "))"   ;
               return result;
            }
         };

         struct mode3
         {
            static inline T process(const T& t0, const T& t1,
                                    const T& t2, const T& t3,
                                    const bfunc_t bf0, const bfunc_t bf1, const bfunc_t bf2)
            {
               // (((T0 o0 T1) o1 T2) o2 T3)
               return bf2(bf1(bf0(t0,t1),t2),t3);
            }

            template <typename T0, typename T1, typename T2, typename T3>
            static inline std::string id()
            {
               static const std::string result = "((" + exprtk_crtype(T0) + "o"    +
                                                        exprtk_crtype(T1) + ")o("  +
                                                        exprtk_crtype(T2) + "))o(" +
                                                        exprtk_crtype(T3) + ")";
               return result;
            }
         };

         struct mode4
         {
            static inline T process(const T& t0, const T& t1,
                                    const T& t2, const T& t3,
                                    const bfunc_t bf0, const bfunc_t bf1, const bfunc_t bf2)
            {
               // ((T0 o0 (T1 o1 T2)) o2 T3
               return bf2(bf0(t0,bf1(t1,t2)),t3);
            }

            template <typename T0, typename T1, typename T2, typename T3>
            static inline std::string id()
            {
               static const std::string result = "((" + exprtk_crtype(T0) + ")o("  +
                                                        exprtk_crtype(T1) + "o"    +
                                                        exprtk_crtype(T2) + "))o(" +
                                                        exprtk_crtype(T3) + ")"    ;
               return result;
            }
         };
      };

      #undef exprtk_crtype

      template <typename T, typename T0, typename T1>
      struct nodetype_T0oT1 { static const typename expression_node<T>::node_type result; };
      template <typename T, typename T0, typename T1>
      const typename expression_node<T>::node_type nodetype_T0oT1<T,T0,T1>::result = expression_node<T>::e_none;

      #define synthesis_node_type_define(T0_, T1_, v_)                                                          \
      template <typename T, typename T0, typename T1>                                                           \
      struct nodetype_T0oT1<T,T0_,T1_> { static const typename expression_node<T>::node_type result; };         \
      template <typename T, typename T0, typename T1>                                                           \
      const typename expression_node<T>::node_type nodetype_T0oT1<T,T0_,T1_>::result = expression_node<T>:: v_; \

      synthesis_node_type_define(const T0&, const T1&,  e_vov)
      synthesis_node_type_define(const T0&, const T1 ,  e_voc)
      synthesis_node_type_define(const T0 , const T1&,  e_cov)
      synthesis_node_type_define(      T0&,       T1&, e_none)
      synthesis_node_type_define(const T0 , const T1 , e_none)
      synthesis_node_type_define(      T0&, const T1 , e_none)
      synthesis_node_type_define(const T0 ,       T1&, e_none)
      synthesis_node_type_define(const T0&,       T1&, e_none)
      synthesis_node_type_define(      T0&, const T1&, e_none)
      #undef synthesis_node_type_define

      template <typename T, typename T0, typename T1, typename T2>
      struct nodetype_T0oT1oT2 { static const typename expression_node<T>::node_type result; };
      template <typename T, typename T0, typename T1, typename T2>
      const typename expression_node<T>::node_type nodetype_T0oT1oT2<T,T0,T1,T2>::result = expression_node<T>::e_none;

      #define synthesis_node_type_define(T0_, T1_, T2_, v_)                                                            \
      template <typename T, typename T0, typename T1, typename T2>                                                     \
      struct nodetype_T0oT1oT2<T,T0_,T1_,T2_> { static const typename expression_node<T>::node_type result; };         \
      template <typename T, typename T0, typename T1, typename T2>                                                     \
      const typename expression_node<T>::node_type nodetype_T0oT1oT2<T,T0_,T1_,T2_>::result = expression_node<T>:: v_; \

      synthesis_node_type_define(const T0&, const T1&, const T2&, e_vovov)
      synthesis_node_type_define(const T0&, const T1&, const T2 , e_vovoc)
      synthesis_node_type_define(const T0&, const T1 , const T2&, e_vocov)
      synthesis_node_type_define(const T0 , const T1&, const T2&, e_covov)
      synthesis_node_type_define(const T0 , const T1&, const T2 , e_covoc)
      synthesis_node_type_define(const T0 , const T1 , const T2 , e_none )
      synthesis_node_type_define(const T0 , const T1 , const T2&, e_none )
      synthesis_node_type_define(const T0&, const T1 , const T2 , e_none )
      synthesis_node_type_define(      T0&,       T1&,       T2&, e_none )
      #undef synthesis_node_type_define

      template <typename T, typename T0, typename T1, typename T2, typename T3>
      struct nodetype_T0oT1oT2oT3 { static const typename expression_node<T>::node_type result; };
      template <typename T, typename T0, typename T1, typename T2, typename T3>
      const typename expression_node<T>::node_type nodetype_T0oT1oT2oT3<T,T0,T1,T2,T3>::result = expression_node<T>::e_none;

      #define synthesis_node_type_define(T0_, T1_, T2_, T3_, v_)                                                              \
      template <typename T, typename T0, typename T1, typename T2, typename T3>                                               \
      struct nodetype_T0oT1oT2oT3<T,T0_,T1_,T2_,T3_> { static const typename expression_node<T>::node_type result; };         \
      template <typename T, typename T0, typename T1, typename T2, typename T3>                                               \
      const typename expression_node<T>::node_type nodetype_T0oT1oT2oT3<T,T0_,T1_,T2_,T3_>::result = expression_node<T>:: v_; \

      synthesis_node_type_define(const T0&, const T1&, const T2&, const T3&, e_vovovov)
      synthesis_node_type_define(const T0&, const T1&, const T2&, const T3 , e_vovovoc)
      synthesis_node_type_define(const T0&, const T1&, const T2 , const T3&, e_vovocov)
      synthesis_node_type_define(const T0&, const T1 , const T2&, const T3&, e_vocovov)
      synthesis_node_type_define(const T0 , const T1&, const T2&, const T3&, e_covovov)
      synthesis_node_type_define(const T0 , const T1&, const T2 , const T3&, e_covocov)
      synthesis_node_type_define(const T0&, const T1 , const T2&, const T3 , e_vocovoc)
      synthesis_node_type_define(const T0 , const T1&, const T2&, const T3 , e_covovoc)
      synthesis_node_type_define(const T0&, const T1 , const T2 , const T3&, e_vococov)
      synthesis_node_type_define(const T0 , const T1 , const T2 , const T3 , e_none   )
      synthesis_node_type_define(const T0 , const T1 , const T2 , const T3&, e_none   )
      synthesis_node_type_define(const T0 , const T1 , const T2&, const T3 , e_none   )
      synthesis_node_type_define(const T0 , const T1&, const T2 , const T3 , e_none   )
      synthesis_node_type_define(const T0&, const T1 , const T2 , const T3 , e_none   )
      synthesis_node_type_define(const T0 , const T1 , const T2&, const T3&, e_none   )
      synthesis_node_type_define(const T0&, const T1&, const T2 , const T3 , e_none   )
      #undef synthesis_node_type_define

      template <typename T, typename T0, typename T1>
      class T0oT1 exprtk_final : public expression_node<T>
      {
      public:

         typedef typename details::functor_t<T> functor_t;
         typedef typename functor_t::bfunc_t    bfunc_t;
         typedef T value_type;
         typedef T0oT1<T,T0,T1> node_type;

         T0oT1(T0 p0, T1 p1, const bfunc_t p2)
         : t0_(p0)
         , t1_(p1)
         , f_ (p2)
         {}

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            static const typename expression_node<T>::node_type result = nodetype_T0oT1<T,T0,T1>::result;
            return result;
         }

         inline operator_type operation() const exprtk_override
         {
            return e_default;
         }

         inline T value() const exprtk_override
         {
            return f_(t0_,t1_);
         }

         inline T0 t0() const
         {
            return t0_;
         }

         inline T1 t1() const
         {
            return t1_;
         }

         inline bfunc_t f() const
         {
            return f_;
         }

         template <typename Allocator>
         static inline expression_node<T>* allocate(Allocator& allocator,
                                                    T0 p0, T1 p1,
                                                    bfunc_t p2)
         {
            return allocator
                     .template allocate_type<node_type, T0, T1, bfunc_t&>
                        (p0, p1, p2);
         }

      private:

         T0oT1(const T0oT1<T,T0,T1>&) exprtk_delete;
         T0oT1<T,T0,T1>& operator=(const T0oT1<T,T0,T1>&) { return (*this); }

         T0 t0_;
         T1 t1_;
         const bfunc_t f_;
      };

      template <typename T, typename T0, typename T1, typename T2, typename ProcessMode>
      class T0oT1oT2 exprtk_final : public T0oT1oT2_base_node<T>
      {
      public:

         typedef typename details::functor_t<T> functor_t;
         typedef typename functor_t::bfunc_t    bfunc_t;
         typedef T value_type;
         typedef T0oT1oT2<T,T0,T1,T2,ProcessMode> node_type;
         typedef ProcessMode process_mode_t;

         T0oT1oT2(T0 p0, T1 p1, T2 p2, const bfunc_t p3, const bfunc_t p4)
         : t0_(p0)
         , t1_(p1)
         , t2_(p2)
         , f0_(p3)
         , f1_(p4)
         {}

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            static const typename expression_node<T>::node_type result = nodetype_T0oT1oT2<T,T0,T1,T2>::result;
            return result;
         }

         inline operator_type operation()
         {
            return e_default;
         }

         inline T value() const exprtk_override
         {
            return ProcessMode::process(t0_, t1_, t2_, f0_, f1_);
         }

         inline T0 t0() const
         {
            return t0_;
         }

         inline T1 t1() const
         {
            return t1_;
         }

         inline T2 t2() const
         {
            return t2_;
         }

         bfunc_t f0() const
         {
            return f0_;
         }

         bfunc_t f1() const
         {
            return f1_;
         }

         std::string type_id() const exprtk_override
         {
            return id();
         }

         static inline std::string id()
         {
            return process_mode_t::template id<T0,T1,T2>();
         }

         template <typename Allocator>
         static inline expression_node<T>* allocate(Allocator& allocator, T0 p0, T1 p1, T2 p2, bfunc_t p3, bfunc_t p4)
         {
            return allocator
                      .template allocate_type<node_type, T0, T1, T2, bfunc_t, bfunc_t>
                         (p0, p1, p2, p3, p4);
         }

      private:

         T0oT1oT2(const node_type&) exprtk_delete;
         node_type& operator=(const node_type&) { return (*this); }

         T0 t0_;
         T1 t1_;
         T2 t2_;
         const bfunc_t f0_;
         const bfunc_t f1_;
      };

      template <typename T, typename T0_, typename T1_, typename T2_, typename T3_, typename ProcessMode>
      class T0oT1oT2oT3 exprtk_final : public T0oT1oT2oT3_base_node<T>
      {
      public:

         typedef typename details::functor_t<T> functor_t;
         typedef typename functor_t::bfunc_t    bfunc_t;
         typedef T value_type;
         typedef T0_ T0;
         typedef T1_ T1;
         typedef T2_ T2;
         typedef T3_ T3;
         typedef T0oT1oT2oT3<T,T0,T1,T2,T3,ProcessMode> node_type;
         typedef ProcessMode process_mode_t;

         T0oT1oT2oT3(T0 p0, T1 p1, T2 p2, T3 p3, bfunc_t p4, bfunc_t p5, bfunc_t p6)
         : t0_(p0)
         , t1_(p1)
         , t2_(p2)
         , t3_(p3)
         , f0_(p4)
         , f1_(p5)
         , f2_(p6)
         {}

         inline T value() const exprtk_override
         {
            return ProcessMode::process(t0_, t1_, t2_, t3_, f0_, f1_, f2_);
         }

         inline T0 t0() const
         {
            return t0_;
         }

         inline T1 t1() const
         {
            return t1_;
         }

         inline T2 t2() const
         {
            return t2_;
         }

         inline T3 t3() const
         {
            return t3_;
         }

         inline bfunc_t f0() const
         {
            return f0_;
         }

         inline bfunc_t f1() const
         {
            return f1_;
         }

         inline bfunc_t f2() const
         {
            return f2_;
         }

         inline std::string type_id() const exprtk_override
         {
            return id();
         }

         static inline std::string id()
         {
            return process_mode_t::template id<T0, T1, T2, T3>();
         }

         template <typename Allocator>
         static inline expression_node<T>* allocate(Allocator& allocator,
                                                    T0 p0, T1 p1, T2 p2, T3 p3,
                                                    bfunc_t p4, bfunc_t p5, bfunc_t p6)
         {
            return allocator
                      .template allocate_type<node_type, T0, T1, T2, T3, bfunc_t, bfunc_t>
                         (p0, p1, p2, p3, p4, p5, p6);
         }

      private:

         T0oT1oT2oT3(const node_type&) exprtk_delete;
         node_type& operator=(const node_type&) { return (*this); }

         T0 t0_;
         T1 t1_;
         T2 t2_;
         T3 t3_;
         const bfunc_t f0_;
         const bfunc_t f1_;
         const bfunc_t f2_;
      };

      template <typename T, typename T0, typename T1, typename T2>
      class T0oT1oT2_sf3 exprtk_final : public T0oT1oT2_base_node<T>
      {
      public:

         typedef typename details::functor_t<T> functor_t;
         typedef typename functor_t::tfunc_t    tfunc_t;
         typedef T value_type;
         typedef T0oT1oT2_sf3<T,T0,T1,T2> node_type;

         T0oT1oT2_sf3(T0 p0, T1 p1, T2 p2, const tfunc_t p3)
         : t0_(p0)
         , t1_(p1)
         , t2_(p2)
         , f_ (p3)
         {}

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            static const typename expression_node<T>::node_type result = nodetype_T0oT1oT2<T,T0,T1,T2>::result;
            return result;
         }

         inline operator_type operation() const exprtk_override
         {
            return e_default;
         }

         inline T value() const exprtk_override
         {
            return f_(t0_, t1_, t2_);
         }

         inline T0 t0() const
         {
            return t0_;
         }

         inline T1 t1() const
         {
            return t1_;
         }

         inline T2 t2() const
         {
            return t2_;
         }

         tfunc_t f() const
         {
            return f_;
         }

         std::string type_id() const
         {
            return id();
         }

         static inline std::string id()
         {
            return "sf3";
         }

         template <typename Allocator>
         static inline expression_node<T>* allocate(Allocator& allocator, T0 p0, T1 p1, T2 p2, tfunc_t p3)
         {
            return allocator
                     .template allocate_type<node_type, T0, T1, T2, tfunc_t>
                        (p0, p1, p2, p3);
         }

      private:

         T0oT1oT2_sf3(const node_type&) exprtk_delete;
         node_type& operator=(const node_type&) { return (*this); }

         T0 t0_;
         T1 t1_;
         T2 t2_;
         const tfunc_t f_;
      };

      template <typename T, typename T0, typename T1, typename T2>
      class sf3ext_type_node : public T0oT1oT2_base_node<T>
      {
      public:

         virtual ~sf3ext_type_node() {}

         virtual T0 t0() const = 0;

         virtual T1 t1() const = 0;

         virtual T2 t2() const = 0;
      };

      template <typename T, typename T0, typename T1, typename T2, typename SF3Operation>
      class T0oT1oT2_sf3ext exprtk_final : public sf3ext_type_node<T,T0,T1,T2>
      {
      public:

         typedef typename details::functor_t<T> functor_t;
         typedef typename functor_t::tfunc_t    tfunc_t;
         typedef T value_type;
         typedef T0oT1oT2_sf3ext<T, T0, T1, T2, SF3Operation> node_type;

         T0oT1oT2_sf3ext(T0 p0, T1 p1, T2 p2)
         : t0_(p0)
         , t1_(p1)
         , t2_(p2)
         {}

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            static const typename expression_node<T>::node_type result = nodetype_T0oT1oT2<T,T0,T1,T2>::result;
            return result;
         }

         inline operator_type operation()
         {
            return e_default;
         }

         inline T value() const exprtk_override
         {
            return SF3Operation::process(t0_, t1_, t2_);
         }

         T0 t0() const exprtk_override
         {
            return t0_;
         }

         T1 t1() const exprtk_override
         {
            return t1_;
         }

         T2 t2() const exprtk_override
         {
            return t2_;
         }

         std::string type_id() const exprtk_override
         {
            return id();
         }

         static inline std::string id()
         {
            return SF3Operation::id();
         }

         template <typename Allocator>
         static inline expression_node<T>* allocate(Allocator& allocator, T0 p0, T1 p1, T2 p2)
         {
            return allocator
                     .template allocate_type<node_type, T0, T1, T2>
                        (p0, p1, p2);
         }

      private:

         T0oT1oT2_sf3ext(const node_type&) exprtk_delete;
         node_type& operator=(const node_type&) { return (*this); }

         T0 t0_;
         T1 t1_;
         T2 t2_;
      };

      template <typename T>
      inline bool is_sf3ext_node(const expression_node<T>* n)
      {
         switch (n->type())
         {
            case expression_node<T>::e_vovov : return true;
            case expression_node<T>::e_vovoc : return true;
            case expression_node<T>::e_vocov : return true;
            case expression_node<T>::e_covov : return true;
            case expression_node<T>::e_covoc : return true;
            default                          : return false;
         }
      }

      template <typename T, typename T0, typename T1, typename T2, typename T3>
      class T0oT1oT2oT3_sf4 exprtk_final : public T0oT1oT2_base_node<T>
      {
      public:

         typedef typename details::functor_t<T> functor_t;
         typedef typename functor_t::qfunc_t    qfunc_t;
         typedef T value_type;
         typedef T0oT1oT2oT3_sf4<T, T0, T1, T2, T3> node_type;

         T0oT1oT2oT3_sf4(T0 p0, T1 p1, T2 p2, T3 p3, const qfunc_t p4)
         : t0_(p0)
         , t1_(p1)
         , t2_(p2)
         , t3_(p3)
         , f_ (p4)
         {}

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            static const typename expression_node<T>::node_type result = nodetype_T0oT1oT2oT3<T,T0,T1,T2,T3>::result;
            return result;
         }

         inline operator_type operation() const exprtk_override
         {
            return e_default;
         }

         inline T value() const exprtk_override
         {
            return f_(t0_, t1_, t2_, t3_);
         }

         inline T0 t0() const
         {
            return t0_;
         }

         inline T1 t1() const
         {
            return t1_;
         }

         inline T2 t2() const
         {
            return t2_;
         }

         inline T3 t3() const
         {
            return t3_;
         }

         qfunc_t f() const
         {
            return f_;
         }

         std::string type_id() const
         {
            return id();
         }

         static inline std::string id()
         {
            return "sf4";
         }

         template <typename Allocator>
         static inline expression_node<T>* allocate(Allocator& allocator, T0 p0, T1 p1, T2 p2, T3 p3, qfunc_t p4)
         {
            return allocator
                     .template allocate_type<node_type, T0, T1, T2, T3, qfunc_t>
                        (p0, p1, p2, p3, p4);
         }

      private:

         T0oT1oT2oT3_sf4(const node_type&) exprtk_delete;
         node_type& operator=(const node_type&) { return (*this); }

         T0 t0_;
         T1 t1_;
         T2 t2_;
         T3 t3_;
         const qfunc_t f_;
      };

      template <typename T, typename T0, typename T1, typename T2, typename T3, typename SF4Operation>
      class T0oT1oT2oT3_sf4ext exprtk_final : public T0oT1oT2oT3_base_node<T>
      {
      public:

         typedef typename details::functor_t<T> functor_t;
         typedef typename functor_t::tfunc_t    tfunc_t;
         typedef T value_type;
         typedef T0oT1oT2oT3_sf4ext<T, T0, T1, T2, T3, SF4Operation> node_type;

         T0oT1oT2oT3_sf4ext(T0 p0, T1 p1, T2 p2, T3 p3)
         : t0_(p0)
         , t1_(p1)
         , t2_(p2)
         , t3_(p3)
         {}

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            static const typename expression_node<T>::node_type result = nodetype_T0oT1oT2oT3<T,T0,T1,T2,T3>::result;
            return result;
         }

         inline T value() const exprtk_override
         {
            return SF4Operation::process(t0_, t1_, t2_, t3_);
         }

         inline T0 t0() const
         {
            return t0_;
         }

         inline T1 t1() const
         {
            return t1_;
         }

         inline T2 t2() const
         {
            return t2_;
         }

         inline T3 t3() const
         {
            return t3_;
         }

         std::string type_id() const exprtk_override
         {
            return id();
         }

         static inline std::string id()
         {
            return SF4Operation::id();
         }

         template <typename Allocator>
         static inline expression_node<T>* allocate(Allocator& allocator, T0 p0, T1 p1, T2 p2, T3 p3)
         {
            return allocator
                     .template allocate_type<node_type, T0, T1, T2, T3>
                        (p0, p1, p2, p3);
         }

      private:

         T0oT1oT2oT3_sf4ext(const node_type&) exprtk_delete;
         node_type& operator=(const node_type&) { return (*this); }

         T0 t0_;
         T1 t1_;
         T2 t2_;
         T3 t3_;
      };

      template <typename T>
      inline bool is_sf4ext_node(const expression_node<T>* n)
      {
         switch (n->type())
         {
            case expression_node<T>::e_vovovov : return true;
            case expression_node<T>::e_vovovoc : return true;
            case expression_node<T>::e_vovocov : return true;
            case expression_node<T>::e_vocovov : return true;
            case expression_node<T>::e_covovov : return true;
            case expression_node<T>::e_covocov : return true;
            case expression_node<T>::e_vocovoc : return true;
            case expression_node<T>::e_covovoc : return true;
            case expression_node<T>::e_vococov : return true;
            default                            : return false;
         }
      }

      template <typename T, typename T0, typename T1>
      struct T0oT1_define
      {
         typedef details::T0oT1<T, T0, T1> type0;
      };

      template <typename T, typename T0, typename T1, typename T2>
      struct T0oT1oT2_define
      {
         typedef details::T0oT1oT2<T, T0, T1, T2, typename T0oT1oT2process<T>::mode0> type0;
         typedef details::T0oT1oT2<T, T0, T1, T2, typename T0oT1oT2process<T>::mode1> type1;
         typedef details::T0oT1oT2_sf3<T, T0, T1, T2> sf3_type;
         typedef details::sf3ext_type_node<T, T0, T1, T2> sf3_type_node;
      };

      template <typename T, typename T0, typename T1, typename T2, typename T3>
      struct T0oT1oT2oT3_define
      {
         typedef details::T0oT1oT2oT3<T, T0, T1, T2, T3, typename T0oT1oT20T3process<T>::mode0> type0;
         typedef details::T0oT1oT2oT3<T, T0, T1, T2, T3, typename T0oT1oT20T3process<T>::mode1> type1;
         typedef details::T0oT1oT2oT3<T, T0, T1, T2, T3, typename T0oT1oT20T3process<T>::mode2> type2;
         typedef details::T0oT1oT2oT3<T, T0, T1, T2, T3, typename T0oT1oT20T3process<T>::mode3> type3;
         typedef details::T0oT1oT2oT3<T, T0, T1, T2, T3, typename T0oT1oT20T3process<T>::mode4> type4;
         typedef details::T0oT1oT2oT3_sf4<T, T0, T1, T2, T3> sf4_type;
      };

      template <typename T, typename Operation>
      class vov_node exprtk_final : public vov_base_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef Operation operation_t;

         // variable op variable node
         explicit vov_node(const T& var0, const T& var1)
         : v0_(var0)
         , v1_(var1)
         {}

         inline T value() const exprtk_override
         {
            return Operation::process(v0_,v1_);
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return Operation::type();
         }

         inline operator_type operation() const exprtk_override
         {
            return Operation::operation();
         }

         inline const T& v0() const exprtk_override
         {
            return v0_;
         }

         inline const T& v1() const exprtk_override
         {
            return v1_;
         }

      protected:

         const T& v0_;
         const T& v1_;

      private:

         vov_node(const vov_node<T,Operation>&) exprtk_delete;
         vov_node<T,Operation>& operator=(const vov_node<T,Operation>&) exprtk_delete;
      };

      template <typename T, typename Operation>
      class cov_node exprtk_final : public cov_base_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef Operation operation_t;

         // constant op variable node
         explicit cov_node(const T& const_var, const T& var)
         : c_(const_var)
         , v_(var)
         {}

         inline T value() const exprtk_override
         {
            return Operation::process(c_,v_);
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return Operation::type();
         }

         inline operator_type operation() const exprtk_override
         {
            return Operation::operation();
         }

         inline const T c() const exprtk_override
         {
            return c_;
         }

         inline const T& v() const exprtk_override
         {
            return v_;
         }

      protected:

         const T  c_;
         const T& v_;

      private:

         cov_node(const cov_node<T,Operation>&) exprtk_delete;
         cov_node<T,Operation>& operator=(const cov_node<T,Operation>&) exprtk_delete;
      };

      template <typename T, typename Operation>
      class voc_node exprtk_final : public voc_base_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef Operation operation_t;

         // variable op constant node
         explicit voc_node(const T& var, const T& const_var)
         : v_(var)
         , c_(const_var)
         {}

         inline T value() const exprtk_override
         {
            return Operation::process(v_,c_);
         }

         inline operator_type operation() const exprtk_override
         {
            return Operation::operation();
         }

         inline const T c() const exprtk_override
         {
            return c_;
         }

         inline const T& v() const exprtk_override
         {
            return v_;
         }

      protected:

         const T& v_;
         const T  c_;

      private:

         voc_node(const voc_node<T,Operation>&) exprtk_delete;
         voc_node<T,Operation>& operator=(const voc_node<T,Operation>&) exprtk_delete;
      };

      template <typename T, typename Operation>
      class vob_node exprtk_final : public vob_base_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef std::pair<expression_ptr,bool> branch_t;
         typedef Operation operation_t;

         // variable op constant node
         explicit vob_node(const T& var, const expression_ptr branch)
         : v_(var)
         {
            construct_branch_pair(branch_, branch);
         }

         inline T value() const exprtk_override
         {
            assert(branch_.first);
            return Operation::process(v_,branch_.first->value());
         }

         inline const T& v() const exprtk_override
         {
            return v_;
         }

         inline expression_node<T>* branch(const std::size_t&) const exprtk_override
         {
            return branch_.first;
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(branch_, node_delete_list);
         }

         std::size_t node_depth() const exprtk_override
         {
            return expression_node<T>::ndb_t::compute_node_depth(branch_);
         }

      private:

         vob_node(const vob_node<T,Operation>&) exprtk_delete;
         vob_node<T,Operation>& operator=(const vob_node<T,Operation>&) exprtk_delete;

         const T& v_;
         branch_t branch_;
      };

      template <typename T, typename Operation>
      class bov_node exprtk_final : public bov_base_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef std::pair<expression_ptr,bool> branch_t;
         typedef Operation operation_t;

         // variable op constant node
         explicit bov_node(const expression_ptr branch, const T& var)
         : v_(var)
         {
            construct_branch_pair(branch_, branch);
         }

         inline T value() const exprtk_override
         {
            assert(branch_.first);
            return Operation::process(branch_.first->value(),v_);
         }

         inline const T& v() const exprtk_override
         {
            return v_;
         }

         inline expression_node<T>* branch(const std::size_t&) const exprtk_override
         {
            return branch_.first;
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(branch_, node_delete_list);
         }

         std::size_t node_depth() const exprtk_override
         {
            return expression_node<T>::ndb_t::compute_node_depth(branch_);
         }

      private:

         bov_node(const bov_node<T,Operation>&) exprtk_delete;
         bov_node<T,Operation>& operator=(const bov_node<T,Operation>&) exprtk_delete;

         const T& v_;
         branch_t branch_;
      };

      template <typename T, typename Operation>
      class cob_node exprtk_final : public cob_base_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef std::pair<expression_ptr,bool> branch_t;
         typedef Operation operation_t;

         // variable op constant node
         explicit cob_node(const T const_var, const expression_ptr branch)
         : c_(const_var)
         {
            construct_branch_pair(branch_, branch);
         }

         inline T value() const exprtk_override
         {
            assert(branch_.first);
            return Operation::process(c_,branch_.first->value());
         }

         inline operator_type operation() const exprtk_override
         {
            return Operation::operation();
         }

         inline const T c() const exprtk_override
         {
            return c_;
         }

         inline void set_c(const T new_c) exprtk_override
         {
            (*const_cast<T*>(&c_)) = new_c;
         }

         inline expression_node<T>* branch(const std::size_t&) const exprtk_override
         {
            return branch_.first;
         }

         inline expression_node<T>* move_branch(const std::size_t&) exprtk_override
         {
            branch_.second = false;
            return branch_.first;
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(branch_, node_delete_list);
         }

         std::size_t node_depth() const exprtk_override
         {
            return expression_node<T>::ndb_t::compute_node_depth(branch_);
         }

      private:

         cob_node(const cob_node<T,Operation>&) exprtk_delete;
         cob_node<T,Operation>& operator=(const cob_node<T,Operation>&) exprtk_delete;

         const T  c_;
         branch_t branch_;
      };

      template <typename T, typename Operation>
      class boc_node exprtk_final : public boc_base_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef std::pair<expression_ptr,bool> branch_t;
         typedef Operation operation_t;

         // variable op constant node
         explicit boc_node(const expression_ptr branch, const T const_var)
         : c_(const_var)
         {
            construct_branch_pair(branch_, branch);
         }

         inline T value() const exprtk_override
         {
            assert(branch_.first);
            return Operation::process(branch_.first->value(),c_);
         }

         inline operator_type operation() const exprtk_override
         {
            return Operation::operation();
         }

         inline const T c() const exprtk_override
         {
            return c_;
         }

         inline void set_c(const T new_c) exprtk_override
         {
            (*const_cast<T*>(&c_)) = new_c;
         }

         inline expression_node<T>* branch(const std::size_t&) const exprtk_override
         {
            return branch_.first;
         }

         inline expression_node<T>* move_branch(const std::size_t&) exprtk_override
         {
            branch_.second = false;
            return branch_.first;
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(branch_, node_delete_list);
         }

         std::size_t node_depth() const exprtk_override
         {
            return expression_node<T>::ndb_t::compute_node_depth(branch_);
         }

      private:

         boc_node(const boc_node<T,Operation>&) exprtk_delete;
         boc_node<T,Operation>& operator=(const boc_node<T,Operation>&) exprtk_delete;

         const T  c_;
         branch_t branch_;
      };

      template <typename T, typename PowOp>
      class ipow_node exprtk_final: public expression_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef PowOp operation_t;

         explicit ipow_node(const T& v)
         : v_(v)
         {}

         inline T value() const exprtk_override
         {
            return PowOp::result(v_);
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_ipow;
         }

      private:

         ipow_node(const ipow_node<T,PowOp>&) exprtk_delete;
         ipow_node<T,PowOp>& operator=(const ipow_node<T,PowOp>&) exprtk_delete;

         const T& v_;
      };

      template <typename T, typename PowOp>
      class bipow_node exprtk_final : public expression_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef std::pair<expression_ptr, bool> branch_t;
         typedef PowOp operation_t;

         explicit bipow_node(expression_ptr branch)
         {
            construct_branch_pair(branch_, branch);
         }

         inline T value() const exprtk_override
         {
            assert(branch_.first);
            return PowOp::result(branch_.first->value());
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_ipow;
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(branch_, node_delete_list);
         }

         std::size_t node_depth() const exprtk_override
         {
            return expression_node<T>::ndb_t::compute_node_depth(branch_);
         }

      private:

         bipow_node(const bipow_node<T,PowOp>&) exprtk_delete;
         bipow_node<T,PowOp>& operator=(const bipow_node<T,PowOp>&) exprtk_delete;

         branch_t branch_;
      };

      template <typename T, typename PowOp>
      class ipowinv_node exprtk_final : public expression_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef PowOp operation_t;

         explicit ipowinv_node(const T& v)
         : v_(v)
         {}

         inline T value() const exprtk_override
         {
            return (T(1) / PowOp::result(v_));
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_ipowinv;
         }

      private:

         ipowinv_node(const ipowinv_node<T,PowOp>&) exprtk_delete;
         ipowinv_node<T,PowOp>& operator=(const ipowinv_node<T,PowOp>&) exprtk_delete;

         const T& v_;
      };

      template <typename T, typename PowOp>
      class bipowninv_node exprtk_final : public expression_node<T>
      {
      public:

         typedef expression_node<T>* expression_ptr;
         typedef std::pair<expression_ptr, bool> branch_t;
         typedef PowOp operation_t;

         explicit bipowninv_node(expression_ptr branch)
         {
            construct_branch_pair(branch_, branch);
         }

         inline T value() const exprtk_override
         {
            assert(branch_.first);
            return (T(1) / PowOp::result(branch_.first->value()));
         }

         inline typename expression_node<T>::node_type type() const exprtk_override
         {
            return expression_node<T>::e_ipowinv;
         }

         void collect_nodes(typename expression_node<T>::noderef_list_t& node_delete_list) exprtk_override
         {
            expression_node<T>::ndb_t::collect(branch_, node_delete_list);
         }

         std::size_t node_depth() const exprtk_override
         {
            return expression_node<T>::ndb_t::compute_node_depth(branch_);
         }

      private:

         bipowninv_node(const bipowninv_node<T,PowOp>&) exprtk_delete;
         bipowninv_node<T,PowOp>& operator=(const bipowninv_node<T,PowOp>&) exprtk_delete;

         branch_t branch_;
      };

      template <typename T>
      inline bool is_vov_node(const expression_node<T>* node)
      {
         return (0 != dynamic_cast<const vov_base_node<T>*>(node));
      }

      template <typename T>
      inline bool is_cov_node(const expression_node<T>* node)
      {
         return (0 != dynamic_cast<const cov_base_node<T>*>(node));
      }

      template <typename T>
      inline bool is_voc_node(const expression_node<T>* node)
      {
         return (0 != dynamic_cast<const voc_base_node<T>*>(node));
      }

      template <typename T>
      inline bool is_cob_node(const expression_node<T>* node)
      {
         return (0 != dynamic_cast<const cob_base_node<T>*>(node));
      }

      template <typename T>
      inline bool is_boc_node(const expression_node<T>* node)
      {
         return (0 != dynamic_cast<const boc_base_node<T>*>(node));
      }

      template <typename T>
      inline bool is_t0ot1ot2_node(const expression_node<T>* node)
      {
         return (0 != dynamic_cast<const T0oT1oT2_base_node<T>*>(node));
      }

      template <typename T>
      inline bool is_t0ot1ot2ot3_node(const expression_node<T>* node)
      {
         return (0 != dynamic_cast<const T0oT1oT2oT3_base_node<T>*>(node));
      }

      template <typename T>
      inline bool is_uv_node(const expression_node<T>* node)
      {
         return (0 != dynamic_cast<const uv_base_node<T>*>(node));
      }

      template <typename T>
      inline bool is_string_node(const expression_node<T>* node)
      {
         return node && (expression_node<T>::e_stringvar == node->type());
      }

      template <typename T>
      inline bool is_string_range_node(const expression_node<T>* node)
      {
         return node && (expression_node<T>::e_stringvarrng == node->type());
      }

      template <typename T>
      inline bool is_const_string_node(const expression_node<T>* node)
      {
         return node && (expression_node<T>::e_stringconst == node->type());
      }

      template <typename T>
      inline bool is_const_string_range_node(const expression_node<T>* node)
      {
         return node && (expression_node<T>::e_cstringvarrng == node->type());
      }

      template <typename T>
      inline bool is_string_assignment_node(const expression_node<T>* node)
      {
         return node && (expression_node<T>::e_strass == node->type());
      }

      template <typename T>
      inline bool is_string_concat_node(const expression_node<T>* node)
      {
         return node && (expression_node<T>::e_strconcat == node->type());
      }

      template <typename T>
      inline bool is_string_function_node(const expression_node<T>* node)
      {
         return node && (expression_node<T>::e_strfunction == node->type());
      }

      template <typename T>
      inline bool is_string_condition_node(const expression_node<T>* node)
      {
         return node && (expression_node<T>::e_strcondition == node->type());
      }

      template <typename T>
      inline bool is_string_ccondition_node(const expression_node<T>* node)
      {
         return node && (expression_node<T>::e_strccondition == node->type());
      }

      template <typename T>
      inline bool is_string_vararg_node(const expression_node<T>* node)
      {
         return node && (expression_node<T>::e_stringvararg == node->type());
      }

      template <typename T>
      inline bool is_genricstring_range_node(const expression_node<T>* node)
      {
         return node && (expression_node<T>::e_strgenrange == node->type());
      }

      template <typename T>
      inline bool is_generally_string_node(const expression_node<T>* node)
      {
         if (node)
         {
            switch (node->type())
            {
               case expression_node<T>::e_stringvar     :
               case expression_node<T>::e_stringconst   :
               case expression_node<T>::e_stringvarrng  :
               case expression_node<T>::e_cstringvarrng :
               case expression_node<T>::e_strgenrange   :
               case expression_node<T>::e_strass        :
               case expression_node<T>::e_strconcat     :
               case expression_node<T>::e_strfunction   :
               case expression_node<T>::e_strcondition  :
               case expression_node<T>::e_strccondition :
               case expression_node<T>::e_stringvararg  : return true;
               default                                  : return false;
            }
         }

         return false;
      }

      class node_allocator
      {
      public:

         template <typename ResultNode, typename OpType, typename ExprNode>
         inline expression_node<typename ResultNode::value_type>* allocate(OpType& operation, ExprNode (&branch)[1])
         {
            expression_node<typename ResultNode::value_type>* result =
               allocate<ResultNode>(operation, branch[0]);
            result->node_depth();
            return result;
         }

         template <typename ResultNode, typename OpType, typename ExprNode>
         inline expression_node<typename ResultNode::value_type>* allocate(OpType& operation, ExprNode (&branch)[2])
         {
            expression_node<typename ResultNode::value_type>* result =
               allocate<ResultNode>(operation, branch[0], branch[1]);
            result->node_depth();
            return result;
         }

         template <typename ResultNode, typename OpType, typename ExprNode>
         inline expression_node<typename ResultNode::value_type>* allocate(OpType& operation, ExprNode (&branch)[3])
         {
            expression_node<typename ResultNode::value_type>* result =
               allocate<ResultNode>(operation, branch[0], branch[1], branch[2]);
            result->node_depth();
            return result;
         }

         template <typename ResultNode, typename OpType, typename ExprNode>
         inline expression_node<typename ResultNode::value_type>* allocate(OpType& operation, ExprNode (&branch)[4])
         {
            expression_node<typename ResultNode::value_type>* result =
               allocate<ResultNode>(operation, branch[0], branch[1], branch[2], branch[3]);
            result->node_depth();
            return result;
         }

         template <typename ResultNode, typename OpType, typename ExprNode>
         inline expression_node<typename ResultNode::value_type>* allocate(OpType& operation, ExprNode (&branch)[5])
         {
            expression_node<typename ResultNode::value_type>* result =
               allocate<ResultNode>(operation, branch[0],branch[1], branch[2], branch[3], branch[4]);
            result->node_depth();
            return result;
         }

         template <typename ResultNode, typename OpType, typename ExprNode>
         inline expression_node<typename ResultNode::value_type>* allocate(OpType& operation, ExprNode (&branch)[6])
         {
            expression_node<typename ResultNode::value_type>* result =
               allocate<ResultNode>(operation, branch[0], branch[1], branch[2], branch[3], branch[4], branch[5]);
            result->node_depth();
            return result;
         }

         template <typename node_type>
         inline expression_node<typename node_type::value_type>* allocate() const
         {
            return (new node_type());
         }

         template <typename node_type,
                   typename Type,
                   typename Allocator,
                   template <typename, typename> class Sequence>
         inline expression_node<typename node_type::value_type>* allocate(const Sequence<Type,Allocator>& seq) const
         {
            expression_node<typename node_type::value_type>*
            result = (new node_type(seq));
            result->node_depth();
            return result;
         }

         template <typename node_type, typename T1>
         inline expression_node<typename node_type::value_type>* allocate(T1& t1) const
         {
            expression_node<typename node_type::value_type>*
            result = (new node_type(t1));
            result->node_depth();
            return result;
         }

         template <typename node_type, typename T1>
         inline expression_node<typename node_type::value_type>* allocate_c(const T1& t1) const
         {
            expression_node<typename node_type::value_type>*
            result = (new node_type(t1));
            result->node_depth();
            return result;
         }

         template <typename node_type,
                   typename T1, typename T2>
         inline expression_node<typename node_type::value_type>* allocate(const T1& t1, const T2& t2) const
         {
            expression_node<typename node_type::value_type>*
            result = (new node_type(t1, t2));
            result->node_depth();
            return result;
         }

         template <typename node_type,
                   typename T1, typename T2>
         inline expression_node<typename node_type::value_type>* allocate_cr(const T1& t1, T2& t2) const
         {
            expression_node<typename node_type::value_type>*
            result = (new node_type(t1, t2));
            result->node_depth();
            return result;
         }

         template <typename node_type,
                   typename T1, typename T2>
         inline expression_node<typename node_type::value_type>* allocate_rc(T1& t1, const T2& t2) const
         {
            expression_node<typename node_type::value_type>*
            result = (new node_type(t1, t2));
            result->node_depth();
            return result;
         }

         template <typename node_type,
                   typename T1, typename T2>
         inline expression_node<typename node_type::value_type>* allocate_rr(T1& t1, T2& t2) const
         {
            expression_node<typename node_type::value_type>*
            result = (new node_type(t1, t2));
            result->node_depth();
            return result;
         }

         template <typename node_type,
                   typename T1, typename T2>
         inline expression_node<typename node_type::value_type>* allocate_tt(T1 t1, T2 t2) const
         {
            expression_node<typename node_type::value_type>*
            result = (new node_type(t1, t2));
            result->node_depth();
            return result;
         }

         template <typename node_type,
                   typename T1, typename T2, typename T3>
         inline expression_node<typename node_type::value_type>* allocate_ttt(T1 t1, T2 t2, T3 t3) const
         {
            expression_node<typename node_type::value_type>*
            result = (new node_type(t1, t2, t3));
            result->node_depth();
            return result;
         }

         template <typename node_type,
                   typename T1, typename T2, typename T3, typename T4>
         inline expression_node<typename node_type::value_type>* allocate_tttt(T1 t1, T2 t2, T3 t3, T4 t4) const
         {
            expression_node<typename node_type::value_type>*
            result = (new node_type(t1, t2, t3, t4));
            result->node_depth();
            return result;
         }

         template <typename node_type,
                   typename T1, typename T2, typename T3>
         inline expression_node<typename node_type::value_type>* allocate_rrr(T1& t1, T2& t2, T3& t3) const
         {
            expression_node<typename node_type::value_type>*
            result = (new node_type(t1, t2, t3));
            result->node_depth();
            return result;
         }

         template <typename node_type,
                   typename T1, typename T2, typename T3, typename T4>
         inline expression_node<typename node_type::value_type>* allocate_rrrr(T1& t1, T2& t2, T3& t3, T4& t4) const
         {
            expression_node<typename node_type::value_type>*
            result = (new node_type(t1, t2, t3, t4));
            result->node_depth();
            return result;
         }

         template <typename node_type,
                   typename T1, typename T2, typename T3, typename T4, typename T5>
         inline expression_node<typename node_type::value_type>* allocate_rrrrr(T1& t1, T2& t2, T3& t3, T4& t4, T5& t5) const
         {
            expression_node<typename node_type::value_type>*
            result = (new node_type(t1, t2, t3, t4, t5));
            result->node_depth();
            return result;
         }

         template <typename node_type,
                   typename T1, typename T2, typename T3>
         inline expression_node<typename node_type::value_type>* allocate(const T1& t1, const T2& t2,
                                                                          const T3& t3) const
         {
            expression_node<typename node_type::value_type>*
            result = (new node_type(t1, t2, t3));
            result->node_depth();
            return result;
         }

         template <typename node_type,
                   typename T1, typename T2,
                   typename T3, typename T4>
         inline expression_node<typename node_type::value_type>* allocate(const T1& t1, const T2& t2,
                                                                          const T3& t3, const T4& t4) const
         {
            expression_node<typename node_type::value_type>*
            result = (new node_type(t1, t2, t3, t4));
            result->node_depth();
            return result;
         }

         template <typename node_type,
                   typename T1, typename T2,
                   typename T3, typename T4, typename T5>
         inline expression_node<typename node_type::value_type>* allocate(const T1& t1, const T2& t2,
                                                                          const T3& t3, const T4& t4,
                                                                          const T5& t5) const
         {
            expression_node<typename node_type::value_type>*
            result = (new node_type(t1, t2, t3, t4, t5));
            result->node_depth();
            return result;
         }

         template <typename node_type,
                   typename T1, typename T2,
                   typename T3, typename T4, typename T5, typename T6>
         inline expression_node<typename node_type::value_type>* allocate(const T1& t1, const T2& t2,
                                                                          const T3& t3, const T4& t4,
                                                                          const T5& t5, const T6& t6) const
         {
            expression_node<typename node_type::value_type>*
            result = (new node_type(t1, t2, t3, t4, t5, t6));
            result->node_depth();
            return result;
         }

         template <typename node_type,
                   typename T1, typename T2,
                   typename T3, typename T4,
                   typename T5, typename T6, typename T7>
         inline expression_node<typename node_type::value_type>* allocate(const T1& t1, const T2& t2,
                                                                          const T3& t3, const T4& t4,
                                                                          const T5& t5, const T6& t6,
                                                                          const T7& t7) const
         {
            expression_node<typename node_type::value_type>*
            result = (new node_type(t1, t2, t3, t4, t5, t6, t7));
            result->node_depth();
            return result;
         }

         template <typename node_type,
                   typename T1, typename T2,
                   typename T3, typename T4,
                   typename T5, typename T6,
                   typename T7, typename T8>
         inline expression_node<typename node_type::value_type>* allocate(const T1& t1, const T2& t2,
                                                                          const T3& t3, const T4& t4,
                                                                          const T5& t5, const T6& t6,
                                                                          const T7& t7, const T8& t8) const
         {
            expression_node<typename node_type::value_type>*
            result = (new node_type(t1, t2, t3, t4, t5, t6, t7, t8));
            result->node_depth();
            return result;
         }

         template <typename node_type,
                   typename T1, typename T2,
                   typename T3, typename T4,
                   typename T5, typename T6,
                   typename T7, typename T8, typename T9>
         inline expression_node<typename node_type::value_type>* allocate(const T1& t1, const T2& t2,
                                                                          const T3& t3, const T4& t4,
                                                                          const T5& t5, const T6& t6,
                                                                          const T7& t7, const T8& t8,
                                                                          const T9& t9) const
         {
            expression_node<typename node_type::value_type>*
            result = (new node_type(t1, t2, t3, t4, t5, t6, t7, t8, t9));
            result->node_depth();
            return result;
         }

         template <typename node_type,
                   typename T1, typename T2,
                   typename T3, typename T4,
                   typename T5, typename T6,
                   typename T7, typename T8,
                   typename T9, typename T10>
         inline expression_node<typename node_type::value_type>* allocate(const T1& t1, const  T2&  t2,
                                                                          const T3& t3, const  T4&  t4,
                                                                          const T5& t5, const  T6&  t6,
                                                                          const T7& t7, const  T8&  t8,
                                                                          const T9& t9, const T10& t10) const
         {
            expression_node<typename node_type::value_type>*
            result = (new node_type(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10));
            result->node_depth();
            return result;
         }

         template <typename node_type,
                   typename T1, typename T2, typename T3>
         inline expression_node<typename node_type::value_type>* allocate_type(T1 t1, T2 t2, T3 t3) const
         {
            expression_node<typename node_type::value_type>*
            result = (new node_type(t1, t2, t3));
            result->node_depth();
            return result;
         }

         template <typename node_type,
                   typename T1, typename T2,
                   typename T3, typename T4>
         inline expression_node<typename node_type::value_type>* allocate_type(T1 t1, T2 t2,
                                                                               T3 t3, T4 t4) const
         {
            expression_node<typename node_type::value_type>*
            result = (new node_type(t1, t2, t3, t4));
            result->node_depth();
            return result;
         }

         template <typename node_type,
                   typename T1, typename T2,
                   typename T3, typename T4,
                   typename T5>
         inline expression_node<typename node_type::value_type>* allocate_type(T1 t1, T2 t2,
                                                                               T3 t3, T4 t4,
                                                                               T5 t5) const
         {
            expression_node<typename node_type::value_type>*
            result = (new node_type(t1, t2, t3, t4, t5));
            result->node_depth();
            return result;
         }

         template <typename node_type,
                   typename T1, typename T2,
                   typename T3, typename T4,
                   typename T5, typename T6>
         inline expression_node<typename node_type::value_type>* allocate_type(T1 t1, T2 t2,
                                                                               T3 t3, T4 t4,
                                                                               T5 t5, T6 t6) const
         {
            expression_node<typename node_type::value_type>*
            result = (new node_type(t1, t2, t3, t4, t5, t6));
            result->node_depth();
            return result;
         }

         template <typename node_type,
                   typename T1, typename T2,
                   typename T3, typename T4,
                   typename T5, typename T6, typename T7>
         inline expression_node<typename node_type::value_type>* allocate_type(T1 t1, T2 t2,
                                                                               T3 t3, T4 t4,
                                                                               T5 t5, T6 t6,
                                                                               T7 t7) const
         {
            expression_node<typename node_type::value_type>*
            result = (new node_type(t1, t2, t3, t4, t5, t6, t7));
            result->node_depth();
            return result;
         }

         template <typename T>
         void inline free(expression_node<T>*& e) const
         {
            exprtk_debug(("node_allocator::free() - deleting expression_node "
                          "type: %03d addr: %p\n",
                          static_cast<int>(e->type()),
                          reinterpret_cast<void*>(e)));
            delete e;
            e = 0;
         }
      };

      inline void load_operations_map(std::multimap<std::string,details::base_operation_t,details::ilesscompare>& m)
      {
         #define register_op(Symbol, Type, Args)                                             \
         m.insert(std::make_pair(std::string(Symbol),details::base_operation_t(Type,Args))); \

         register_op("abs"       , e_abs     , 1)
         register_op("acos"      , e_acos    , 1)
         register_op("acosh"     , e_acosh   , 1)
         register_op("asin"      , e_asin    , 1)
         register_op("asinh"     , e_asinh   , 1)
         register_op("atan"      , e_atan    , 1)
         register_op("atanh"     , e_atanh   , 1)
         register_op("ceil"      , e_ceil    , 1)
         register_op("cos"       , e_cos     , 1)
         register_op("cosh"      , e_cosh    , 1)
         register_op("exp"       , e_exp     , 1)
         register_op("expm1"     , e_expm1   , 1)
         register_op("floor"     , e_floor   , 1)
         register_op("log"       , e_log     , 1)
         register_op("log10"     , e_log10   , 1)
         register_op("log2"      , e_log2    , 1)
         register_op("log1p"     , e_log1p   , 1)
         register_op("round"     , e_round   , 1)
         register_op("sin"       , e_sin     , 1)
         register_op("sinc"      , e_sinc    , 1)
         register_op("sinh"      , e_sinh    , 1)
         register_op("sec"       , e_sec     , 1)
         register_op("csc"       , e_csc     , 1)
         register_op("sqrt"      , e_sqrt    , 1)
         register_op("tan"       , e_tan     , 1)
         register_op("tanh"      , e_tanh    , 1)
         register_op("cot"       , e_cot     , 1)
         register_op("rad2deg"   , e_r2d     , 1)
         register_op("deg2rad"   , e_d2r     , 1)
         register_op("deg2grad"  , e_d2g     , 1)
         register_op("grad2deg"  , e_g2d     , 1)
         register_op("sgn"       , e_sgn     , 1)
         register_op("not"       , e_notl    , 1)
         register_op("erf"       , e_erf     , 1)
         register_op("erfc"      , e_erfc    , 1)
         register_op("ncdf"      , e_ncdf    , 1)
         register_op("frac"      , e_frac    , 1)
         register_op("trunc"     , e_trunc   , 1)
         register_op("atan2"     , e_atan2   , 2)
         register_op("mod"       , e_mod     , 2)
         register_op("logn"      , e_logn    , 2)
         register_op("pow"       , e_pow     , 2)
         register_op("root"      , e_root    , 2)
         register_op("roundn"    , e_roundn  , 2)
         register_op("equal"     , e_equal   , 2)
         register_op("not_equal" , e_nequal  , 2)
         register_op("hypot"     , e_hypot   , 2)
         register_op("shr"       , e_shr     , 2)
         register_op("shl"       , e_shl     , 2)
         register_op("clamp"     , e_clamp   , 3)
         register_op("iclamp"    , e_iclamp  , 3)
         register_op("inrange"   , e_inrange , 3)
         #undef register_op
      }

   } // namespace details

   class function_traits
   {
   public:

      function_traits()
      : allow_zero_parameters_(false)
      , has_side_effects_(true)
      , min_num_args_(0)
      , max_num_args_(std::numeric_limits<std::size_t>::max())
      {}

      inline bool& allow_zero_parameters()
      {
         return allow_zero_parameters_;
      }

      inline bool& has_side_effects()
      {
         return has_side_effects_;
      }

      std::size_t& min_num_args()
      {
         return min_num_args_;
      }

      std::size_t& max_num_args()
      {
         return max_num_args_;
      }

   private:

      bool allow_zero_parameters_;
      bool has_side_effects_;
      std::size_t min_num_args_;
      std::size_t max_num_args_;
   };

   template <typename FunctionType>
   void enable_zero_parameters(FunctionType& func)
   {
      func.allow_zero_parameters() = true;

      if (0 != func.min_num_args())
      {
         func.min_num_args() = 0;
      }
   }

   template <typename FunctionType>
   void disable_zero_parameters(FunctionType& func)
   {
      func.allow_zero_parameters() = false;
   }

   template <typename FunctionType>
   void enable_has_side_effects(FunctionType& func)
   {
      func.has_side_effects() = true;
   }

   template <typename FunctionType>
   void disable_has_side_effects(FunctionType& func)
   {
      func.has_side_effects() = false;
   }

   template <typename FunctionType>
   void set_min_num_args(FunctionType& func, const std::size_t& num_args)
   {
      func.min_num_args() = num_args;

      if ((0 != func.min_num_args()) && func.allow_zero_parameters())
         func.allow_zero_parameters() = false;
   }

   template <typename FunctionType>
   void set_max_num_args(FunctionType& func, const std::size_t& num_args)
   {
      func.max_num_args() = num_args;
   }

   template <typename T>
   class ifunction : public function_traits
   {
   public:

      explicit ifunction(const std::size_t& pc)
      : param_count(pc)
      {}

      virtual ~ifunction() {}

      #define empty_method_body(N)                   \
      {                                              \
         exprtk_debug(("ifunction::operator() - Operator(" #N ") has not been overridden\n")); \
         return std::numeric_limits<T>::quiet_NaN(); \
      }                                              \

      inline virtual T operator() ()
      empty_method_body(0)

      inline virtual T operator() (const T&)
      empty_method_body(1)

      inline virtual T operator() (const T&,const T&)
      empty_method_body(2)

      inline virtual T operator() (const T&, const T&, const T&)
      empty_method_body(3)

      inline virtual T operator() (const T&, const T&, const T&, const T&)
      empty_method_body(4)

      inline virtual T operator() (const T&, const T&, const T&, const T&, const T&)
      empty_method_body(5)

      inline virtual T operator() (const T&, const T&, const T&, const T&, const T&, const T&)
      empty_method_body(6)

      inline virtual T operator() (const T&, const T&, const T&, const T&, const T&, const T&, const T&)
      empty_method_body(7)

      inline virtual T operator() (const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&)
      empty_method_body(8)

      inline virtual T operator() (const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&)
      empty_method_body(9)

      inline virtual T operator() (const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&)
      empty_method_body(10)

      inline virtual T operator() (const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&,
                                   const T&)
      empty_method_body(11)

      inline virtual T operator() (const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&,
                                   const T&, const T&)
      empty_method_body(12)

      inline virtual T operator() (const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&,
                                   const T&, const T&, const T&)
      empty_method_body(13)

      inline virtual T operator() (const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&,
                                   const T&, const T&, const T&, const T&)
      empty_method_body(14)

      inline virtual T operator() (const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&,
                                   const T&, const T&, const T&, const T&, const T&)
      empty_method_body(15)

      inline virtual T operator() (const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&,
                                   const T&, const T&, const T&, const T&, const T&, const T&)
      empty_method_body(16)

      inline virtual T operator() (const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&,
                                   const T&, const T&, const T&, const T&, const T&, const T&, const T&)
      empty_method_body(17)

      inline virtual T operator() (const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&,
                                   const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&)
      empty_method_body(18)

      inline virtual T operator() (const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&,
                                   const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&)
      empty_method_body(19)

      inline virtual T operator() (const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&,
                                   const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&)
      empty_method_body(20)

      #undef empty_method_body

      std::size_t param_count;
   };

   template <typename T>
   class ivararg_function : public function_traits
   {
   public:

      virtual ~ivararg_function() {}

      inline virtual T operator() (const std::vector<T>&)
      {
         exprtk_debug(("ivararg_function::operator() - Operator has not been overridden\n"));
         return std::numeric_limits<T>::quiet_NaN();
      }
   };

   template <typename T>
   class igeneric_function : public function_traits
   {
   public:

      enum return_type
      {
         e_rtrn_scalar   = 0,
         e_rtrn_string   = 1,
         e_rtrn_overload = 2
      };

      typedef T type;
      typedef type_store<T> generic_type;
      typedef typename generic_type::parameter_list parameter_list_t;

      igeneric_function(const std::string& param_seq = "", const return_type rtr_type = e_rtrn_scalar)
      : parameter_sequence(param_seq)
      , rtrn_type(rtr_type)
      {}

      virtual ~igeneric_function() {}

      #define igeneric_function_empty_body(N)        \
      {                                              \
         exprtk_debug(("igeneric_function::operator() - Operator(" #N ") has not been overridden\n")); \
         return std::numeric_limits<T>::quiet_NaN(); \
      }                                              \

      // f(i_0,i_1,....,i_N) --> Scalar
      inline virtual T operator() (parameter_list_t)
      igeneric_function_empty_body(1)

      // f(i_0,i_1,....,i_N) --> String
      inline virtual T operator() (std::string&, parameter_list_t)
      igeneric_function_empty_body(2)

      // f(psi,i_0,i_1,....,i_N) --> Scalar
      inline virtual T operator() (const std::size_t&, parameter_list_t)
      igeneric_function_empty_body(3)

      // f(psi,i_0,i_1,....,i_N) --> String
      inline virtual T operator() (const std::size_t&, std::string&, parameter_list_t)
      igeneric_function_empty_body(4)

      std::string parameter_sequence;
      return_type rtrn_type;
   };

   template <typename T> class parser;
   template <typename T> class expression_helper;

   template <typename T>
   class symbol_table
   {
   public:

      typedef T (*ff00_functor)();
      typedef T (*ff01_functor)(T);
      typedef T (*ff02_functor)(T, T);
      typedef T (*ff03_functor)(T, T, T);
      typedef T (*ff04_functor)(T, T, T, T);
      typedef T (*ff05_functor)(T, T, T, T, T);
      typedef T (*ff06_functor)(T, T, T, T, T, T);
      typedef T (*ff07_functor)(T, T, T, T, T, T, T);
      typedef T (*ff08_functor)(T, T, T, T, T, T, T, T);
      typedef T (*ff09_functor)(T, T, T, T, T, T, T, T, T);
      typedef T (*ff10_functor)(T, T, T, T, T, T, T, T, T, T);
      typedef T (*ff11_functor)(T, T, T, T, T, T, T, T, T, T, T);
      typedef T (*ff12_functor)(T, T, T, T, T, T, T, T, T, T, T, T);
      typedef T (*ff13_functor)(T, T, T, T, T, T, T, T, T, T, T, T, T);
      typedef T (*ff14_functor)(T, T, T, T, T, T, T, T, T, T, T, T, T, T);
      typedef T (*ff15_functor)(T, T, T, T, T, T, T, T, T, T, T, T, T, T, T);

   protected:

       struct freefunc00 : public exprtk::ifunction<T>
       {
          using exprtk::ifunction<T>::operator();

          explicit freefunc00(ff00_functor ff) : exprtk::ifunction<T>(0), f(ff) {}
          inline T operator() ()
          { return f(); }
          ff00_functor f;
       };

      struct freefunc01 : public exprtk::ifunction<T>
      {
         using exprtk::ifunction<T>::operator();

         explicit freefunc01(ff01_functor ff) : exprtk::ifunction<T>(1), f(ff) {}
         inline T operator() (const T& v0)
         { return f(v0); }
         ff01_functor f;
      };

      struct freefunc02 : public exprtk::ifunction<T>
      {
         using exprtk::ifunction<T>::operator();

         explicit freefunc02(ff02_functor ff) : exprtk::ifunction<T>(2), f(ff) {}
         inline T operator() (const T& v0, const T& v1)
         { return f(v0, v1); }
         ff02_functor f;
      };

      struct freefunc03 : public exprtk::ifunction<T>
      {
         using exprtk::ifunction<T>::operator();

         explicit freefunc03(ff03_functor ff) : exprtk::ifunction<T>(3), f(ff) {}
         inline T operator() (const T& v0, const T& v1, const T& v2)
         { return f(v0, v1, v2); }
         ff03_functor f;
      };

      struct freefunc04 : public exprtk::ifunction<T>
      {
         using exprtk::ifunction<T>::operator();

         explicit freefunc04(ff04_functor ff) : exprtk::ifunction<T>(4), f(ff) {}
         inline T operator() (const T& v0, const T& v1, const T& v2, const T& v3)
         { return f(v0, v1, v2, v3); }
         ff04_functor f;
      };

      struct freefunc05 : public exprtk::ifunction<T>
      {
         using exprtk::ifunction<T>::operator();

         explicit freefunc05(ff05_functor ff) : exprtk::ifunction<T>(5), f(ff) {}
         inline T operator() (const T& v0, const T& v1, const T& v2, const T& v3, const T& v4)
         { return f(v0, v1, v2, v3, v4); }
         ff05_functor f;
      };

      struct freefunc06 : public exprtk::ifunction<T>
      {
         using exprtk::ifunction<T>::operator();

         explicit freefunc06(ff06_functor ff) : exprtk::ifunction<T>(6), f(ff) {}
         inline T operator() (const T& v0, const T& v1, const T& v2, const T& v3, const T& v4, const T& v5)
         { return f(v0, v1, v2, v3, v4, v5); }
         ff06_functor f;
      };

      struct freefunc07 : public exprtk::ifunction<T>
      {
         using exprtk::ifunction<T>::operator();

         explicit freefunc07(ff07_functor ff) : exprtk::ifunction<T>(7), f(ff) {}
         inline T operator() (const T& v0, const T& v1, const T& v2, const T& v3, const T& v4,
                              const T& v5, const T& v6)
         { return f(v0, v1, v2, v3, v4, v5, v6); }
         ff07_functor f;
      };

      struct freefunc08 : public exprtk::ifunction<T>
      {
         using exprtk::ifunction<T>::operator();

         explicit freefunc08(ff08_functor ff) : exprtk::ifunction<T>(8), f(ff) {}
         inline T operator() (const T& v0, const T& v1, const T& v2, const T& v3, const T& v4,
                              const T& v5, const T& v6, const T& v7)
         { return f(v0, v1, v2, v3, v4, v5, v6, v7); }
         ff08_functor f;
      };

      struct freefunc09 : public exprtk::ifunction<T>
      {
         using exprtk::ifunction<T>::operator();

         explicit freefunc09(ff09_functor ff) : exprtk::ifunction<T>(9), f(ff) {}
         inline T operator() (const T& v0, const T& v1, const T& v2, const T& v3, const T& v4,
                              const T& v5, const T& v6, const T& v7, const T& v8)
         { return f(v0, v1, v2, v3, v4, v5, v6, v7, v8); }
         ff09_functor f;
      };

      struct freefunc10 : public exprtk::ifunction<T>
      {
         using exprtk::ifunction<T>::operator();

         explicit freefunc10(ff10_functor ff) : exprtk::ifunction<T>(10), f(ff) {}
         inline T operator() (const T& v0, const T& v1, const T& v2, const T& v3, const T& v4,
                              const T& v5, const T& v6, const T& v7, const T& v8, const T& v9)
         { return f(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9); }
         ff10_functor f;
      };

      struct freefunc11 : public exprtk::ifunction<T>
      {
         using exprtk::ifunction<T>::operator();

         explicit freefunc11(ff11_functor ff) : exprtk::ifunction<T>(11), f(ff) {}
         inline T operator() (const T& v0, const T& v1, const T& v2, const T& v3, const T& v4,
                              const T& v5, const T& v6, const T& v7, const T& v8, const T& v9, const T& v10)
         { return f(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10); }
         ff11_functor f;
      };

      struct freefunc12 : public exprtk::ifunction<T>
      {
         using exprtk::ifunction<T>::operator();

         explicit freefunc12(ff12_functor ff) : exprtk::ifunction<T>(12), f(ff) {}
         inline T operator() (const T& v00, const T& v01, const T& v02, const T& v03, const T& v04,
                              const T& v05, const T& v06, const T& v07, const T& v08, const T& v09,
                              const T& v10, const T& v11)
         { return f(v00, v01, v02, v03, v04, v05, v06, v07, v08, v09, v10, v11); }
         ff12_functor f;
      };

      struct freefunc13 : public exprtk::ifunction<T>
      {
         using exprtk::ifunction<T>::operator();

         explicit freefunc13(ff13_functor ff) : exprtk::ifunction<T>(13), f(ff) {}
         inline T operator() (const T& v00, const T& v01, const T& v02, const T& v03, const T& v04,
                              const T& v05, const T& v06, const T& v07, const T& v08, const T& v09,
                              const T& v10, const T& v11, const T& v12)
         { return f(v00, v01, v02, v03, v04, v05, v06, v07, v08, v09, v10, v11, v12); }
         ff13_functor f;
      };

      struct freefunc14 : public exprtk::ifunction<T>
      {
         using exprtk::ifunction<T>::operator();

         explicit freefunc14(ff14_functor ff) : exprtk::ifunction<T>(14), f(ff) {}
         inline T operator() (const T& v00, const T& v01, const T& v02, const T& v03, const T& v04,
                              const T& v05, const T& v06, const T& v07, const T& v08, const T& v09,
                              const T& v10, const T& v11, const T& v12, const T& v13)
         { return f(v00, v01, v02, v03, v04, v05, v06, v07, v08, v09, v10, v11, v12, v13); }
         ff14_functor f;
      };

      struct freefunc15 : public exprtk::ifunction<T>
      {
         using exprtk::ifunction<T>::operator();

         explicit freefunc15(ff15_functor ff) : exprtk::ifunction<T>(15), f(ff) {}
         inline T operator() (const T& v00, const T& v01, const T& v02, const T& v03, const T& v04,
                              const T& v05, const T& v06, const T& v07, const T& v08, const T& v09,
                              const T& v10, const T& v11, const T& v12, const T& v13, const T& v14)
         { return f(v00, v01, v02, v03, v04, v05, v06, v07, v08, v09, v10, v11, v12, v13, v14); }
         ff15_functor f;
      };

      template <typename Type, typename RawType>
      struct type_store
      {
         typedef details::expression_node<T>*        expression_ptr;
         typedef typename details::variable_node<T>  variable_node_t;
         typedef ifunction<T>                        ifunction_t;
         typedef ivararg_function<T>                 ivararg_function_t;
         typedef igeneric_function<T>                igeneric_function_t;
         typedef details::vector_holder<T>           vector_t;

         typedef Type type_t;
         typedef type_t* type_ptr;
         typedef std::pair<bool,type_ptr> type_pair_t;
         typedef std::map<std::string,type_pair_t,details::ilesscompare> type_map_t;
         typedef typename type_map_t::iterator tm_itr_t;
         typedef typename type_map_t::const_iterator tm_const_itr_t;

         enum { lut_size = 256 };

         type_map_t  map;
         std::size_t size;

         type_store()
         : size(0)
         {}

         struct deleter
         {
            #define exprtk_define_process(Type)                  \
            static inline void process(std::pair<bool,Type*>& n) \
            {                                                    \
               delete n.second;                                  \
            }                                                    \

            exprtk_define_process(variable_node_t )
            exprtk_define_process(vector_t        )

            #undef exprtk_define_process

            template <typename DeleteType>
            static inline void process(std::pair<bool,DeleteType*>&)
            {}
         };

         inline bool symbol_exists(const std::string& symbol_name) const
         {
            if (symbol_name.empty())
               return false;
            else if (map.end() != map.find(symbol_name))
               return true;
            else
               return false;
         }

         template <typename PtrType>
         inline std::string entity_name(const PtrType& ptr) const
         {
            if (map.empty())
               return std::string();

            tm_const_itr_t itr = map.begin();

            while (map.end() != itr)
            {
               if (itr->second.second == ptr)
               {
                  return itr->first;
               }
               else
                  ++itr;
            }

            return std::string();
         }

         inline bool is_constant(const std::string& symbol_name) const
         {
            if (symbol_name.empty())
               return false;
            else
            {
               const tm_const_itr_t itr = map.find(symbol_name);

               if (map.end() == itr)
                  return false;
               else
                  return (*itr).second.first;
            }
         }

         template <typename Tie, typename RType>
         inline bool add_impl(const std::string& symbol_name, RType t, const bool is_const)
         {
            if (symbol_name.size() > 1)
            {
               for (std::size_t i = 0; i < details::reserved_symbols_size; ++i)
               {
                  if (details::imatch(symbol_name, details::reserved_symbols[i]))
                  {
                     return false;
                  }
               }
            }

            const tm_itr_t itr = map.find(symbol_name);

            if (map.end() == itr)
            {
               map[symbol_name] = Tie::make(t,is_const);
               ++size;
            }

            return true;
         }

         struct tie_array
         {
            static inline std::pair<bool,vector_t*> make(std::pair<T*,std::size_t> v, const bool is_const = false)
            {
               return std::make_pair(is_const, new vector_t(v.first, v.second));
            }
         };

         struct tie_stdvec
         {
            template <typename Allocator>
            static inline std::pair<bool,vector_t*> make(std::vector<T,Allocator>& v, const bool is_const = false)
            {
               return std::make_pair(is_const, new vector_t(v));
            }
         };

         struct tie_vecview
         {
            static inline std::pair<bool,vector_t*> make(exprtk::vector_view<T>& v, const bool is_const = false)
            {
               return std::make_pair(is_const, new vector_t(v));
            }
         };

         struct tie_stddeq
         {
            template <typename Allocator>
            static inline std::pair<bool,vector_t*> make(std::deque<T,Allocator>& v, const bool is_const = false)
            {
               return std::make_pair(is_const, new vector_t(v));
            }
         };

         template <std::size_t v_size>
         inline bool add(const std::string& symbol_name, T (&v)[v_size], const bool is_const = false)
         {
            return add_impl<tie_array,std::pair<T*,std::size_t> >
                      (symbol_name, std::make_pair(v,v_size), is_const);
         }

         inline bool add(const std::string& symbol_name, T* v, const std::size_t v_size, const bool is_const = false)
         {
            return add_impl<tie_array,std::pair<T*,std::size_t> >
                     (symbol_name, std::make_pair(v,v_size), is_const);
         }

         template <typename Allocator>
         inline bool add(const std::string& symbol_name, std::vector<T,Allocator>& v, const bool is_const = false)
         {
            return add_impl<tie_stdvec,std::vector<T,Allocator>&>
                      (symbol_name, v, is_const);
         }

         inline bool add(const std::string& symbol_name, exprtk::vector_view<T>& v, const bool is_const = false)
         {
            return add_impl<tie_vecview,exprtk::vector_view<T>&>
                      (symbol_name, v, is_const);
         }

         template <typename Allocator>
         inline bool add(const std::string& symbol_name, std::deque<T,Allocator>& v, const bool is_const = false)
         {
            return add_impl<tie_stddeq,std::deque<T,Allocator>&>
                      (symbol_name, v, is_const);
         }

         inline bool add(const std::string& symbol_name, RawType& t_, const bool is_const = false)
         {
            struct tie
            {
               static inline std::pair<bool,variable_node_t*> make(T& t, const bool is_constant = false)
               {
                  return std::make_pair(is_constant, new variable_node_t(t));
               }

               static inline std::pair<bool,function_t*> make(function_t& t, const bool is_constant = false)
               {
                  return std::make_pair(is_constant,&t);
               }

               static inline std::pair<bool,vararg_function_t*> make(vararg_function_t& t, const bool is_constant = false)
               {
                  return std::make_pair(is_constant,&t);
               }

               static inline std::pair<bool,generic_function_t*> make(generic_function_t& t, const bool is_constant = false)
               {
                  return std::make_pair(is_constant,&t);
               }
            };

            const tm_itr_t itr = map.find(symbol_name);

            if (map.end() == itr)
            {
               map[symbol_name] = tie::make(t_,is_const);
               ++size;
            }

            return true;
         }

         inline type_ptr get(const std::string& symbol_name) const
         {
            const tm_const_itr_t itr = map.find(symbol_name);

            if (map.end() == itr)
               return reinterpret_cast<type_ptr>(0);
            else
               return itr->second.second;
         }

         template <typename TType, typename TRawType, typename PtrType>
         struct ptr_match
         {
            static inline bool test(const PtrType, const void*)
            {
               return false;
            }
         };

         template <typename TType, typename TRawType>
         struct ptr_match<TType,TRawType,variable_node_t*>
         {
            static inline bool test(const variable_node_t* p, const void* ptr)
            {
               exprtk_debug(("ptr_match::test() - %p <--> %p\n",(void*)(&(p->ref())),ptr));
               return (&(p->ref()) == ptr);
            }
         };

         inline type_ptr get_from_varptr(const void* ptr) const
         {
            tm_const_itr_t itr = map.begin();

            while (map.end() != itr)
            {
               type_ptr ret_ptr = itr->second.second;

               if (ptr_match<Type,RawType,type_ptr>::test(ret_ptr,ptr))
               {
                  return ret_ptr;
               }

               ++itr;
            }

            return type_ptr(0);
         }

         inline bool remove(const std::string& symbol_name, const bool delete_node = true)
         {
            const tm_itr_t itr = map.find(symbol_name);

            if (map.end() != itr)
            {
               if (delete_node)
               {
                  deleter::process((*itr).second);
               }

               map.erase(itr);
               --size;

               return true;
            }
            else
               return false;
         }

         inline RawType& type_ref(const std::string& symbol_name)
         {
            struct init_type
            {
               static inline double set(double)           { return (0.0);           }
               static inline double set(long double)      { return (0.0);           }
               static inline float  set(float)            { return (0.0f);          }
               static inline std::string set(std::string) { return std::string(""); }
            };

            static RawType null_type = init_type::set(RawType());

            const tm_const_itr_t itr = map.find(symbol_name);

            if (map.end() == itr)
               return null_type;
            else
               return itr->second.second->ref();
         }

         inline void clear(const bool delete_node = true)
         {
            if (!map.empty())
            {
               if (delete_node)
               {
                  tm_itr_t itr = map.begin();
                  tm_itr_t end = map.end  ();

                  while (end != itr)
                  {
                     deleter::process((*itr).second);
                     ++itr;
                  }
               }

               map.clear();
            }

            size = 0;
         }

         template <typename Allocator,
                   template <typename, typename> class Sequence>
         inline std::size_t get_list(Sequence<std::pair<std::string,RawType>,Allocator>& list) const
         {
            std::size_t count = 0;

            if (!map.empty())
            {
               tm_const_itr_t itr = map.begin();
               tm_const_itr_t end = map.end  ();

               while (end != itr)
               {
                  list.push_back(std::make_pair((*itr).first,itr->second.second->ref()));
                  ++itr;
                  ++count;
               }
            }

            return count;
         }

         template <typename Allocator,
                   template <typename, typename> class Sequence>
         inline std::size_t get_list(Sequence<std::string,Allocator>& vlist) const
         {
            std::size_t count = 0;

            if (!map.empty())
            {
               tm_const_itr_t itr = map.begin();
               tm_const_itr_t end = map.end  ();

               while (end != itr)
               {
                  vlist.push_back((*itr).first);
                  ++itr;
                  ++count;
               }
            }

            return count;
         }
      };

      typedef details::expression_node<T>*        expression_ptr;
      typedef typename details::variable_node<T>  variable_t;
      typedef typename details::vector_holder<T>  vector_holder_t;
      typedef variable_t*                         variable_ptr;
      typedef ifunction        <T>                function_t;
      typedef ivararg_function <T>                vararg_function_t;
      typedef igeneric_function<T>                generic_function_t;
      typedef function_t*                         function_ptr;
      typedef vararg_function_t*                  vararg_function_ptr;
      typedef generic_function_t*                 generic_function_ptr;

      static const std::size_t lut_size = 256;

      // Symbol Table Holder
      struct control_block
      {
         struct st_data
         {
            type_store<variable_t        , T                 > variable_store;
            type_store<function_t        , function_t        > function_store;
            type_store<vararg_function_t , vararg_function_t > vararg_function_store;
            type_store<generic_function_t, generic_function_t> generic_function_store;
            type_store<generic_function_t, generic_function_t> string_function_store;
            type_store<generic_function_t, generic_function_t> overload_function_store;
            type_store<vector_holder_t   , vector_holder_t   > vector_store;

            st_data()
            {
               for (std::size_t i = 0; i < details::reserved_words_size; ++i)
               {
                  reserved_symbol_table_.insert(details::reserved_words[i]);
               }

               for (std::size_t i = 0; i < details::reserved_symbols_size; ++i)
               {
                  reserved_symbol_table_.insert(details::reserved_symbols[i]);
               }
            }

           ~st_data()
            {
               for (std::size_t i = 0; i < free_function_list_.size(); ++i)
               {
                  delete free_function_list_[i];
               }
            }

            inline bool is_reserved_symbol(const std::string& symbol) const
            {
               return (reserved_symbol_table_.end() != reserved_symbol_table_.find(symbol));
            }

            static inline st_data* create()
            {
               return (new st_data);
            }

            static inline void destroy(st_data*& sd)
            {
               delete sd;
               sd = reinterpret_cast<st_data*>(0);
            }

            std::list<T>               local_symbol_list_;
            std::list<std::string>     local_stringvar_list_;
            std::set<std::string>      reserved_symbol_table_;
            std::vector<ifunction<T>*> free_function_list_;
         };

         control_block()
         : ref_count(1)
         , data_(st_data::create())
         {}

         explicit control_block(st_data* data)
         : ref_count(1)
         , data_(data)
         {}

        ~control_block()
         {
            if (data_ && (0 == ref_count))
            {
               st_data::destroy(data_);
            }
         }

         static inline control_block* create()
         {
            return (new control_block);
         }

         template <typename SymTab>
         static inline void destroy(control_block*& cntrl_blck, SymTab* sym_tab)
         {
            if (cntrl_blck)
            {
               if (
                    (0 !=   cntrl_blck->ref_count) &&
                    (0 == --cntrl_blck->ref_count)
                  )
               {
                  if (sym_tab)
                     sym_tab->clear();

                  delete cntrl_blck;
               }

               cntrl_blck = 0;
            }
         }

         std::size_t ref_count;
         st_data* data_;
      };

   public:

      symbol_table()
      : control_block_(control_block::create())
      {
         clear();
      }

     ~symbol_table()
      {
         control_block::destroy(control_block_,this);
      }

      symbol_table(const symbol_table<T>& st)
      {
         control_block_ = st.control_block_;
         control_block_->ref_count++;
      }

      inline symbol_table<T>& operator=(const symbol_table<T>& st)
      {
         if (this != &st)
         {
            control_block::destroy(control_block_,reinterpret_cast<symbol_table<T>*>(0));

            control_block_ = st.control_block_;
            control_block_->ref_count++;
         }

         return (*this);
      }

      inline bool operator==(const symbol_table<T>& st) const
      {
         return (this == &st) || (control_block_ == st.control_block_);
      }

      inline void clear_variables(const bool delete_node = true)
      {
         local_data().variable_store.clear(delete_node);
      }

      inline void clear_functions()
      {
         local_data().function_store.clear();
      }

      inline void clear_strings()
      {
      }

      inline void clear_vectors()
      {
         local_data().vector_store.clear();
      }

      inline void clear_local_constants()
      {
         local_data().local_symbol_list_.clear();
      }

      inline void clear()
      {
         if (!valid()) return;
         clear_variables      ();
         clear_functions      ();
         clear_strings        ();
         clear_vectors        ();
         clear_local_constants();
      }

      inline std::size_t variable_count() const
      {
         if (valid())
            return local_data().variable_store.size;
         else
            return 0;
      }

      inline std::size_t function_count() const
      {
         if (valid())
            return local_data().function_store.size;
         else
            return 0;
      }

      inline std::size_t vector_count() const
      {
         if (valid())
            return local_data().vector_store.size;
         else
            return 0;
      }

      inline variable_ptr get_variable(const std::string& variable_name) const
      {
         if (!valid())
            return reinterpret_cast<variable_ptr>(0);
         else if (!valid_symbol(variable_name))
            return reinterpret_cast<variable_ptr>(0);
         else
            return local_data().variable_store.get(variable_name);
      }

      inline variable_ptr get_variable(const T& var_ref) const
      {
         if (!valid())
            return reinterpret_cast<variable_ptr>(0);
         else
            return local_data().variable_store.get_from_varptr(
                                                  reinterpret_cast<const void*>(&var_ref));
      }

      inline function_ptr get_function(const std::string& function_name) const
      {
         if (!valid())
            return reinterpret_cast<function_ptr>(0);
         else if (!valid_symbol(function_name))
            return reinterpret_cast<function_ptr>(0);
         else
            return local_data().function_store.get(function_name);
      }

      inline vararg_function_ptr get_vararg_function(const std::string& vararg_function_name) const
      {
         if (!valid())
            return reinterpret_cast<vararg_function_ptr>(0);
         else if (!valid_symbol(vararg_function_name))
            return reinterpret_cast<vararg_function_ptr>(0);
         else
            return local_data().vararg_function_store.get(vararg_function_name);
      }

      inline generic_function_ptr get_generic_function(const std::string& function_name) const
      {
         if (!valid())
            return reinterpret_cast<generic_function_ptr>(0);
         else if (!valid_symbol(function_name))
            return reinterpret_cast<generic_function_ptr>(0);
         else
            return local_data().generic_function_store.get(function_name);
      }

      inline generic_function_ptr get_string_function(const std::string& function_name) const
      {
         if (!valid())
            return reinterpret_cast<generic_function_ptr>(0);
         else if (!valid_symbol(function_name))
            return reinterpret_cast<generic_function_ptr>(0);
         else
            return local_data().string_function_store.get(function_name);
      }

      inline generic_function_ptr get_overload_function(const std::string& function_name) const
      {
         if (!valid())
            return reinterpret_cast<generic_function_ptr>(0);
         else if (!valid_symbol(function_name))
            return reinterpret_cast<generic_function_ptr>(0);
         else
            return local_data().overload_function_store.get(function_name);
      }

      typedef vector_holder_t* vector_holder_ptr;

      inline vector_holder_ptr get_vector(const std::string& vector_name) const
      {
         if (!valid())
            return reinterpret_cast<vector_holder_ptr>(0);
         else if (!valid_symbol(vector_name))
            return reinterpret_cast<vector_holder_ptr>(0);
         else
            return local_data().vector_store.get(vector_name);
      }

      inline T& variable_ref(const std::string& symbol_name)
      {
         static T null_var = T(0);
         if (!valid())
            return null_var;
         else if (!valid_symbol(symbol_name))
            return null_var;
         else
            return local_data().variable_store.type_ref(symbol_name);
      }

      inline bool is_constant_node(const std::string& symbol_name) const
      {
         if (!valid())
            return false;
         else if (!valid_symbol(symbol_name))
            return false;
         else
            return local_data().variable_store.is_constant(symbol_name);
      }

      inline bool create_variable(const std::string& variable_name, const T& value = T(0))
      {
         if (!valid())
            return false;
         else if (!valid_symbol(variable_name))
            return false;
         else if (symbol_exists(variable_name))
            return false;

         local_data().local_symbol_list_.push_back(value);
         T& t = local_data().local_symbol_list_.back();

         return add_variable(variable_name,t);
      }

      inline bool add_variable(const std::string& variable_name, T& t, const bool is_constant = false)
      {
         if (!valid())
            return false;
         else if (!valid_symbol(variable_name))
            return false;
         else if (symbol_exists(variable_name))
            return false;
         else
            return local_data().variable_store.add(variable_name, t, is_constant);
      }

      inline bool add_constant(const std::string& constant_name, const T& value)
      {
         if (!valid())
            return false;
         else if (!valid_symbol(constant_name))
            return false;
         else if (symbol_exists(constant_name))
            return false;

         local_data().local_symbol_list_.push_back(value);
         T& t = local_data().local_symbol_list_.back();

         return add_variable(constant_name, t, true);
      }

      inline bool add_function(const std::string& function_name, function_t& function)
      {
         if (!valid())
            return false;
         else if (!valid_symbol(function_name))
            return false;
         else if (symbol_exists(function_name))
            return false;
         else
            return local_data().function_store.add(function_name,function);
      }

      inline bool add_function(const std::string& vararg_function_name, vararg_function_t& vararg_function)
      {
         if (!valid())
            return false;
         else if (!valid_symbol(vararg_function_name))
            return false;
         else if (symbol_exists(vararg_function_name))
            return false;
         else
            return local_data().vararg_function_store.add(vararg_function_name,vararg_function);
      }

      inline bool add_function(const std::string& function_name, generic_function_t& function)
      {
         if (!valid())
            return false;
         else if (!valid_symbol(function_name))
            return false;
         else if (symbol_exists(function_name))
            return false;
         else
         {
            switch (function.rtrn_type)
            {
               case generic_function_t::e_rtrn_scalar :
                  return (std::string::npos == function.parameter_sequence.find_first_not_of("STVZ*?|")) ?
                         local_data().generic_function_store.add(function_name,function) : false;

               case generic_function_t::e_rtrn_string :
                  return (std::string::npos == function.parameter_sequence.find_first_not_of("STVZ*?|")) ?
                         local_data().string_function_store.add(function_name,function)  : false;

               case generic_function_t::e_rtrn_overload :
                  return (std::string::npos == function.parameter_sequence.find_first_not_of("STVZ*?|:")) ?
                         local_data().overload_function_store.add(function_name,function) : false;
            }
         }

         return false;
      }

      #define exprtk_define_freefunction(NN)                                                \
      inline bool add_function(const std::string& function_name, ff##NN##_functor function) \
      {                                                                                     \
         if (!valid())                                                                      \
         { return false; }                                                                  \
         if (!valid_symbol(function_name))                                                  \
         { return false; }                                                                  \
         if (symbol_exists(function_name))                                                  \
         { return false; }                                                                  \
                                                                                            \
         exprtk::ifunction<T>* ifunc = new freefunc##NN(function);                          \
                                                                                            \
         local_data().free_function_list_.push_back(ifunc);                                 \
                                                                                            \
         return add_function(function_name,(*local_data().free_function_list_.back()));     \
      }                                                                                     \

      exprtk_define_freefunction(00) exprtk_define_freefunction(01)
      exprtk_define_freefunction(02) exprtk_define_freefunction(03)
      exprtk_define_freefunction(04) exprtk_define_freefunction(05)
      exprtk_define_freefunction(06) exprtk_define_freefunction(07)
      exprtk_define_freefunction(08) exprtk_define_freefunction(09)
      exprtk_define_freefunction(10) exprtk_define_freefunction(11)
      exprtk_define_freefunction(12) exprtk_define_freefunction(13)
      exprtk_define_freefunction(14) exprtk_define_freefunction(15)

      #undef exprtk_define_freefunction

      inline bool add_reserved_function(const std::string& function_name, function_t& function)
      {
         if (!valid())
            return false;
         else if (!valid_symbol(function_name,false))
            return false;
         else if (symbol_exists(function_name,false))
            return false;
         else
            return local_data().function_store.add(function_name,function);
      }

      inline bool add_reserved_function(const std::string& vararg_function_name, vararg_function_t& vararg_function)
      {
         if (!valid())
            return false;
         else if (!valid_symbol(vararg_function_name,false))
            return false;
         else if (symbol_exists(vararg_function_name,false))
            return false;
         else
            return local_data().vararg_function_store.add(vararg_function_name,vararg_function);
      }

      inline bool add_reserved_function(const std::string& function_name, generic_function_t& function)
      {
         if (!valid())
            return false;
         else if (!valid_symbol(function_name,false))
            return false;
         else if (symbol_exists(function_name,false))
            return false;
         else
         {
            switch (function.rtrn_type)
            {
               case generic_function_t::e_rtrn_scalar :
                  return (std::string::npos == function.parameter_sequence.find_first_not_of("STVZ*?|")) ?
                         local_data().generic_function_store.add(function_name,function) : false;

               case generic_function_t::e_rtrn_string :
                  return (std::string::npos == function.parameter_sequence.find_first_not_of("STVZ*?|")) ?
                         local_data().string_function_store.add(function_name,function)  : false;

               case generic_function_t::e_rtrn_overload :
                  return (std::string::npos == function.parameter_sequence.find_first_not_of("STVZ*?|:")) ?
                         local_data().overload_function_store.add(function_name,function) : false;
            }
         }

         return false;
      }

      template <std::size_t N>
      inline bool add_vector(const std::string& vector_name, T (&v)[N])
      {
         if (!valid())
            return false;
         else if (!valid_symbol(vector_name))
            return false;
         else if (symbol_exists(vector_name))
            return false;
         else
            return local_data().vector_store.add(vector_name,v);
      }

      inline bool add_vector(const std::string& vector_name, T* v, const std::size_t& v_size)
      {
         if (!valid())
            return false;
         else if (!valid_symbol(vector_name))
            return false;
         else if (symbol_exists(vector_name))
            return false;
         else if (0 == v_size)
            return false;
         else
            return local_data().vector_store.add(vector_name, v, v_size);
      }

      template <typename Allocator>
      inline bool add_vector(const std::string& vector_name, std::vector<T,Allocator>& v)
      {
         if (!valid())
            return false;
         else if (!valid_symbol(vector_name))
            return false;
         else if (symbol_exists(vector_name))
            return false;
         else if (0 == v.size())
            return false;
         else
            return local_data().vector_store.add(vector_name,v);
      }

      inline bool add_vector(const std::string& vector_name, exprtk::vector_view<T>& v)
      {
         if (!valid())
            return false;
         else if (!valid_symbol(vector_name))
            return false;
         else if (symbol_exists(vector_name))
            return false;
         else if (0 == v.size())
            return false;
         else
            return local_data().vector_store.add(vector_name,v);
      }

      inline bool remove_variable(const std::string& variable_name, const bool delete_node = true)
      {
         if (!valid())
            return false;
         else
            return local_data().variable_store.remove(variable_name, delete_node);
      }

      inline bool remove_function(const std::string& function_name)
      {
         if (!valid())
            return false;
         else
            return local_data().function_store.remove(function_name);
      }

      inline bool remove_vararg_function(const std::string& vararg_function_name)
      {
         if (!valid())
            return false;
         else
            return local_data().vararg_function_store.remove(vararg_function_name);
      }

      inline bool remove_vector(const std::string& vector_name)
      {
         if (!valid())
            return false;
         else
            return local_data().vector_store.remove(vector_name);
      }

      inline bool add_constants()
      {
         return add_pi      () &&
                add_epsilon () &&
                add_infinity() ;
      }

      inline bool add_pi()
      {
         const typename details::numeric::details::number_type<T>::type num_type;
         static const T local_pi = details::numeric::details::const_pi_impl<T>(num_type);
         return add_constant("pi",local_pi);
      }

      inline bool add_epsilon()
      {
         static const T local_epsilon = details::numeric::details::epsilon_type<T>::value();
         return add_constant("epsilon",local_epsilon);
      }

      inline bool add_infinity()
      {
         static const T local_infinity = std::numeric_limits<T>::infinity();
         return add_constant("inf",local_infinity);
      }

      template <typename Package>
      inline bool add_package(Package& package)
      {
         return package.register_package(*this);
      }

      template <typename Allocator,
                template <typename, typename> class Sequence>
      inline std::size_t get_variable_list(Sequence<std::pair<std::string,T>,Allocator>& vlist) const
      {
         if (!valid())
            return 0;
         else
            return local_data().variable_store.get_list(vlist);
      }

      template <typename Allocator,
                template <typename, typename> class Sequence>
      inline std::size_t get_variable_list(Sequence<std::string,Allocator>& vlist) const
      {
         if (!valid())
            return 0;
         else
            return local_data().variable_store.get_list(vlist);
      }

      template <typename Allocator,
                template <typename, typename> class Sequence>
      inline std::size_t get_vector_list(Sequence<std::string,Allocator>& vlist) const
      {
         if (!valid())
            return 0;
         else
            return local_data().vector_store.get_list(vlist);
      }

      inline bool symbol_exists(const std::string& symbol_name, const bool check_reserved_symb = true) const
      {
         /*
            Function will return true if symbol_name exists as either a
            reserved symbol, variable, stringvar, vector or function name
            in any of the type stores.
         */
         if (!valid())
            return false;
         else if (local_data().variable_store.symbol_exists(symbol_name))
            return true;
         else if (local_data().vector_store.symbol_exists(symbol_name))
            return true;
         else if (local_data().function_store.symbol_exists(symbol_name))
            return true;
         else if (check_reserved_symb && local_data().is_reserved_symbol(symbol_name))
            return true;
         else
            return false;
      }

      inline bool is_variable(const std::string& variable_name) const
      {
         if (!valid())
            return false;
         else
            return local_data().variable_store.symbol_exists(variable_name);
      }

      inline bool is_function(const std::string& function_name) const
      {
         if (!valid())
            return false;
         else
            return local_data().function_store.symbol_exists(function_name);
      }

      inline bool is_vararg_function(const std::string& vararg_function_name) const
      {
         if (!valid())
            return false;
         else
            return local_data().vararg_function_store.symbol_exists(vararg_function_name);
      }

      inline bool is_vector(const std::string& vector_name) const
      {
         if (!valid())
            return false;
         else
            return local_data().vector_store.symbol_exists(vector_name);
      }

      inline std::string get_variable_name(const expression_ptr& ptr) const
      {
         return local_data().variable_store.entity_name(ptr);
      }

      inline std::string get_vector_name(const vector_holder_ptr& ptr) const
      {
         return local_data().vector_store.entity_name(ptr);
      }

      inline bool valid() const
      {
         // Symbol table sanity check.
         return control_block_ && control_block_->data_;
      }

      inline void load_from(const symbol_table<T>& st)
      {
         {
            std::vector<std::string> name_list;

            st.local_data().function_store.get_list(name_list);

            if (!name_list.empty())
            {
               for (std::size_t i = 0; i < name_list.size(); ++i)
               {
                  exprtk::ifunction<T>& ifunc = *st.get_function(name_list[i]);
                  add_function(name_list[i],ifunc);
               }
            }
         }

         {
            std::vector<std::string> name_list;

            st.local_data().vararg_function_store.get_list(name_list);

            if (!name_list.empty())
            {
               for (std::size_t i = 0; i < name_list.size(); ++i)
               {
                  exprtk::ivararg_function<T>& ivafunc = *st.get_vararg_function(name_list[i]);
                  add_function(name_list[i],ivafunc);
               }
            }
         }

         {
            std::vector<std::string> name_list;

            st.local_data().generic_function_store.get_list(name_list);

            if (!name_list.empty())
            {
               for (std::size_t i = 0; i < name_list.size(); ++i)
               {
                  exprtk::igeneric_function<T>& ifunc = *st.get_generic_function(name_list[i]);
                  add_function(name_list[i],ifunc);
               }
            }
         }

         {
            std::vector<std::string> name_list;

            st.local_data().string_function_store.get_list(name_list);

            if (!name_list.empty())
            {
               for (std::size_t i = 0; i < name_list.size(); ++i)
               {
                  exprtk::igeneric_function<T>& ifunc = *st.get_string_function(name_list[i]);
                  add_function(name_list[i],ifunc);
               }
            }
         }

         {
            std::vector<std::string> name_list;

            st.local_data().overload_function_store.get_list(name_list);

            if (!name_list.empty())
            {
               for (std::size_t i = 0; i < name_list.size(); ++i)
               {
                  exprtk::igeneric_function<T>& ifunc = *st.get_overload_function(name_list[i]);
                  add_function(name_list[i],ifunc);
               }
            }
         }
      }

   private:

      inline bool valid_symbol(const std::string& symbol, const bool check_reserved_symb = true) const
      {
         if (symbol.empty())
            return false;
         else if (!details::is_letter(symbol[0]))
            return false;
         else if (symbol.size() > 1)
         {
            for (std::size_t i = 1; i < symbol.size(); ++i)
            {
               if (
                    !details::is_letter_or_digit(symbol[i]) &&
                    ('_' != symbol[i])
                  )
               {
                  if ((i < (symbol.size() - 1)) && ('.' == symbol[i]))
                     continue;
                  else
                     return false;
               }
            }
         }

         return (check_reserved_symb) ? (!local_data().is_reserved_symbol(symbol)) : true;
      }

      inline bool valid_function(const std::string& symbol) const
      {
         if (symbol.empty())
            return false;
         else if (!details::is_letter(symbol[0]))
            return false;
         else if (symbol.size() > 1)
         {
            for (std::size_t i = 1; i < symbol.size(); ++i)
            {
               if (
                    !details::is_letter_or_digit(symbol[i]) &&
                    ('_' != symbol[i])
                  )
               {
                  if ((i < (symbol.size() - 1)) && ('.' == symbol[i]))
                     continue;
                  else
                     return false;
               }
            }
         }

         return true;
      }

      typedef typename control_block::st_data local_data_t;

      inline local_data_t& local_data()
      {
         return *(control_block_->data_);
      }

      inline const local_data_t& local_data() const
      {
         return *(control_block_->data_);
      }

      control_block* control_block_;

      friend class parser<T>;
   }; // class symbol_table

   template <typename T>
   class function_compositor;

   template <typename T>
   class expression
   {
   private:

      typedef details::expression_node<T>*  expression_ptr;
      typedef details::vector_holder<T>*    vector_holder_ptr;
      typedef std::vector<symbol_table<T> > symtab_list_t;

      struct control_block
      {
         enum data_type
         {
            e_unknown  ,
            e_expr     ,
            e_vecholder,
            e_data     ,
            e_vecdata  ,
            e_string
         };

         struct data_pack
         {
            data_pack()
            : pointer(0)
            , type(e_unknown)
            , size(0)
            {}

            data_pack(void* ptr, const data_type dt, const std::size_t sz = 0)
            : pointer(ptr)
            , type(dt)
            , size(sz)
            {}

            void*       pointer;
            data_type   type;
            std::size_t size;
         };

         typedef std::vector<data_pack> local_data_list_t;
         typedef results_context<T>     results_context_t;
         typedef control_block*         cntrl_blck_ptr_t;

         control_block()
         : ref_count(0)
         , expr     (0)
         , results  (0)
         , retinv_null(false)
         , return_invoked(&retinv_null)
         {}

         explicit control_block(expression_ptr e)
         : ref_count(1)
         , expr     (e)
         , results  (0)
         , retinv_null(false)
         , return_invoked(&retinv_null)
         {}

        ~control_block()
         {
            if (expr && details::branch_deletable(expr))
            {
               destroy_node(expr);
            }

            if (!local_data_list.empty())
            {
               for (std::size_t i = 0; i < local_data_list.size(); ++i)
               {
                  switch (local_data_list[i].type)
                  {
                     case e_expr      : delete reinterpret_cast<expression_ptr>(local_data_list[i].pointer);
                                        break;

                     case e_vecholder : delete reinterpret_cast<vector_holder_ptr>(local_data_list[i].pointer);
                                        break;

                     case e_data      : delete reinterpret_cast<T*>(local_data_list[i].pointer);
                                        break;

                     case e_vecdata   : delete [] reinterpret_cast<T*>(local_data_list[i].pointer);
                                        break;

                     case e_string    : delete reinterpret_cast<std::string*>(local_data_list[i].pointer);
                                        break;

                     default          : break;
                  }
               }
            }

            if (results)
            {
               delete results;
            }
         }

         static inline cntrl_blck_ptr_t create(expression_ptr e)
         {
            return new control_block(e);
         }

         static inline void destroy(cntrl_blck_ptr_t& cntrl_blck)
         {
            if (cntrl_blck)
            {
               if (
                    (0 !=   cntrl_blck->ref_count) &&
                    (0 == --cntrl_blck->ref_count)
                  )
               {
                  delete cntrl_blck;
               }

               cntrl_blck = 0;
            }
         }

         std::size_t ref_count;
         expression_ptr expr;
         local_data_list_t local_data_list;
         results_context_t* results;
         bool  retinv_null;
         bool* return_invoked;

         friend class function_compositor<T>;
      };

   public:

      expression()
      : control_block_(0)
      {
         set_expression(new details::null_node<T>());
      }

      expression(const expression<T>& e)
      : control_block_    (e.control_block_    )
      , symbol_table_list_(e.symbol_table_list_)
      {
         control_block_->ref_count++;
      }

      explicit expression(const symbol_table<T>& symbol_table)
      : control_block_(0)
      {
         set_expression(new details::null_node<T>());
         symbol_table_list_.push_back(symbol_table);
      }

      inline expression<T>& operator=(const expression<T>& e)
      {
         if (this != &e)
         {
            if (control_block_)
            {
               if (
                    (0 !=   control_block_->ref_count) &&
                    (0 == --control_block_->ref_count)
                  )
               {
                  delete control_block_;
               }

               control_block_ = 0;
            }

            control_block_ = e.control_block_;
            control_block_->ref_count++;
            symbol_table_list_ = e.symbol_table_list_;
         }

         return *this;
      }

      inline bool operator==(const expression<T>& e) const
      {
         return (this == &e);
      }

      inline bool operator!() const
      {
         return (
                  (0 == control_block_      ) ||
                  (0 == control_block_->expr)
                );
      }

      inline expression<T>& release()
      {
         control_block::destroy(control_block_);

         return (*this);
      }

     ~expression()
      {
         control_block::destroy(control_block_);
      }

      inline T value() const
      {
         assert(control_block_      );
         assert(control_block_->expr);

         return control_block_->expr->value();
      }

      inline T operator() () const
      {
         return value();
      }

      inline operator T() const
      {
         return value();
      }

      inline operator bool() const
      {
         return details::is_true(value());
      }

      inline void register_symbol_table(symbol_table<T>& st)
      {
         symbol_table_list_.push_back(st);
      }

      inline const symbol_table<T>& get_symbol_table(const std::size_t& index = 0) const
      {
         return symbol_table_list_[index];
      }

      inline symbol_table<T>& get_symbol_table(const std::size_t& index = 0)
      {
         return symbol_table_list_[index];
      }

      typedef results_context<T> results_context_t;

      inline const results_context_t& results() const
      {
         if (control_block_->results)
            return (*control_block_->results);
         else
         {
            static const results_context_t null_results;
            return null_results;
         }
      }

      inline bool return_invoked() const
      {
         return (*control_block_->return_invoked);
      }

   private:

      inline symtab_list_t get_symbol_table_list() const
      {
         return symbol_table_list_;
      }

      inline void set_expression(const expression_ptr expr)
      {
         if (expr)
         {
            if (control_block_)
            {
               if (0 == --control_block_->ref_count)
               {
                  delete control_block_;
               }
            }

            control_block_ = control_block::create(expr);
         }
      }

      inline void register_local_var(expression_ptr expr)
      {
         if (expr)
         {
            if (control_block_)
            {
               control_block_->
                  local_data_list.push_back(
                     typename expression<T>::control_block::
                        data_pack(reinterpret_cast<void*>(expr),
                                  control_block::e_expr));
            }
         }
      }

      inline void register_local_var(vector_holder_ptr vec_holder)
      {
         if (vec_holder)
         {
            if (control_block_)
            {
               control_block_->
                  local_data_list.push_back(
                     typename expression<T>::control_block::
                        data_pack(reinterpret_cast<void*>(vec_holder),
                                  control_block::e_vecholder));
            }
         }
      }

      inline void register_local_data(void* data, const std::size_t& size = 0, const std::size_t data_mode = 0)
      {
         if (data)
         {
            if (control_block_)
            {
               typename control_block::data_type dt = control_block::e_data;

               switch (data_mode)
               {
                  case 0 : dt = control_block::e_data;    break;
                  case 1 : dt = control_block::e_vecdata; break;
                  case 2 : dt = control_block::e_string;  break;
               }

               control_block_->
                  local_data_list.push_back(
                     typename expression<T>::control_block::
                        data_pack(reinterpret_cast<void*>(data), dt, size));
            }
         }
      }

      inline const typename control_block::local_data_list_t& local_data_list()
      {
         if (control_block_)
         {
            return control_block_->local_data_list;
         }
         else
         {
            static typename control_block::local_data_list_t null_local_data_list;
            return null_local_data_list;
         }
      }

      inline void register_return_results(results_context_t* rc)
      {
         if (control_block_ && rc)
         {
            control_block_->results = rc;
         }
      }

      inline void set_retinvk(bool* retinvk_ptr)
      {
         if (control_block_)
         {
            control_block_->return_invoked = retinvk_ptr;
         }
      }

      control_block* control_block_;
      symtab_list_t  symbol_table_list_;

      friend class parser<T>;
      friend class expression_helper<T>;
      friend class function_compositor<T>;
   }; // class expression

   template <typename T>
   class expression_helper
   {
   public:

      static inline bool is_constant(const expression<T>& expr)
      {
         return details::is_constant_node(expr.control_block_->expr);
      }

      static inline bool is_variable(const expression<T>& expr)
      {
         return details::is_variable_node(expr.control_block_->expr);
      }

      static inline bool is_unary(const expression<T>& expr)
      {
         return details::is_unary_node(expr.control_block_->expr);
      }

      static inline bool is_binary(const expression<T>& expr)
      {
         return details::is_binary_node(expr.control_block_->expr);
      }

      static inline bool is_function(const expression<T>& expr)
      {
         return details::is_function(expr.control_block_->expr);
      }

      static inline bool is_null(const expression<T>& expr)
      {
         return details::is_null_node(expr.control_block_->expr);
      }
   };

   template <typename T>
   inline bool is_valid(const expression<T>& expr)
   {
      return !expression_helper<T>::is_null(expr);
   }

   namespace parser_error
   {
      enum error_mode
      {
         e_unknown = 0,
         e_syntax  = 1,
         e_token   = 2,
         e_numeric = 4,
         e_symtab  = 5,
         e_lexer   = 6,
         e_helper  = 7,
         e_parser  = 8
      };

      struct type
      {
         type()
         : mode(parser_error::e_unknown)
         , line_no  (0)
         , column_no(0)
         {}

         lexer::token token;
         error_mode mode;
         std::string diagnostic;
         std::string src_location;
         std::string error_line;
         std::size_t line_no;
         std::size_t column_no;
      };

      inline type make_error(const error_mode mode,
                             const std::string& diagnostic   = "",
                             const std::string& src_location = "")
      {
         type t;
         t.mode         = mode;
         t.token.type   = lexer::token::e_error;
         t.diagnostic   = diagnostic;
         t.src_location = src_location;
         exprtk_debug(("%s\n",diagnostic .c_str()));
         return t;
      }

      inline type make_error(const error_mode mode,
                             const lexer::token& tk,
                             const std::string& diagnostic   = "",
                             const std::string& src_location = "")
      {
         type t;
         t.mode         = mode;
         t.token        = tk;
         t.diagnostic   = diagnostic;
         t.src_location = src_location;
         exprtk_debug(("%s\n",diagnostic .c_str()));
         return t;
      }

      inline std::string to_str(error_mode mode)
      {
         switch (mode)
         {
            case e_unknown : return std::string("Unknown Error");
            case e_syntax  : return std::string("Syntax Error" );
            case e_token   : return std::string("Token Error"  );
            case e_numeric : return std::string("Numeric Error");
            case e_symtab  : return std::string("Symbol Error" );
            case e_lexer   : return std::string("Lexer Error"  );
            case e_helper  : return std::string("Helper Error" );
            case e_parser  : return std::string("Parser Error" );
            default        : return std::string("Unknown Error");
         }
      }

      inline bool update_error(type& error, const std::string& expression)
      {
         if (
              expression.empty()                         ||
              (error.token.position > expression.size()) ||
              (std::numeric_limits<std::size_t>::max() == error.token.position)
            )
         {
            return false;
         }

         std::size_t error_line_start = 0;

         for (std::size_t i = error.token.position; i > 0; --i)
         {
            const details::char_t c = expression[i];

            if (('\n' == c) || ('\r' == c))
            {
               error_line_start = i + 1;
               break;
            }
         }

         std::size_t next_nl_position = std::min(expression.size(),
                                                 expression.find_first_of('\n',error.token.position + 1));

         error.column_no  = error.token.position - error_line_start;
         error.error_line = expression.substr(error_line_start,
                                              next_nl_position - error_line_start);

         error.line_no = 0;

         for (std::size_t i = 0; i < next_nl_position; ++i)
         {
            if ('\n' == expression[i])
               ++error.line_no;
         }

         return true;
      }

      inline void dump_error(const type& error)
      {
         printf("Position: %02d   Type: [%s]   Msg: %s\n",
                static_cast<int>(error.token.position),
                exprtk::parser_error::to_str(error.mode).c_str(),
                error.diagnostic.c_str());
      }
   }

   namespace details
   {
      template <typename Parser>
      inline void disable_type_checking(Parser& p)
      {
         p.state_.type_check_enabled = false;
      }
   }

   template <typename T>
   class parser : public lexer::parser_helper
   {
   private:

      enum precedence_level
      {
         e_level00, e_level01, e_level02, e_level03, e_level04,
         e_level05, e_level06, e_level07, e_level08, e_level09,
         e_level10, e_level11, e_level12, e_level13, e_level14
      };

      typedef const T&                                    cref_t;
      typedef const T                                     const_t;
      typedef ifunction<T>                                F;
      typedef ivararg_function<T>                         VAF;
      typedef igeneric_function<T>                        GF;
      typedef ifunction<T>                                ifunction_t;
      typedef ivararg_function<T>                         ivararg_function_t;
      typedef igeneric_function<T>                        igeneric_function_t;
      typedef details::expression_node<T>                 expression_node_t;
      typedef details::literal_node<T>                    literal_node_t;
      typedef details::unary_node<T>                      unary_node_t;
      typedef details::binary_node<T>                     binary_node_t;
      typedef details::trinary_node<T>                    trinary_node_t;
      typedef details::quaternary_node<T>                 quaternary_node_t;
      typedef details::conditional_node<T>                conditional_node_t;
      typedef details::cons_conditional_node<T>           cons_conditional_node_t;
      typedef details::while_loop_node<T>                 while_loop_node_t;
      typedef details::repeat_until_loop_node<T>          repeat_until_loop_node_t;
      typedef details::for_loop_node<T>                   for_loop_node_t;
      typedef details::while_loop_rtc_node<T>             while_loop_rtc_node_t;
      typedef details::repeat_until_loop_rtc_node<T>      repeat_until_loop_rtc_node_t;
      typedef details::for_loop_rtc_node<T>               for_loop_rtc_node_t;
      typedef details::switch_node<T>                     switch_node_t;
      typedef details::variable_node<T>                   variable_node_t;
      typedef details::vector_elem_node<T>                vector_elem_node_t;
      typedef details::rebasevector_elem_node<T>          rebasevector_elem_node_t;
      typedef details::rebasevector_celem_node<T>         rebasevector_celem_node_t;
      typedef details::vector_node<T>                     vector_node_t;
      typedef details::range_pack<T>                      range_t;
      typedef details::assignment_node<T>                 assignment_node_t;
      typedef details::assignment_vec_elem_node<T>        assignment_vec_elem_node_t;
      typedef details::assignment_rebasevec_elem_node<T>  assignment_rebasevec_elem_node_t;
      typedef details::assignment_rebasevec_celem_node<T> assignment_rebasevec_celem_node_t;
      typedef details::assignment_vec_node<T>             assignment_vec_node_t;
      typedef details::assignment_vecvec_node<T>          assignment_vecvec_node_t;
      typedef details::conditional_vector_node<T>         conditional_vector_node_t;
      typedef details::scand_node<T>                      scand_node_t;
      typedef details::scor_node<T>                       scor_node_t;
      typedef lexer::token                                token_t;
      typedef expression_node_t*                          expression_node_ptr;
      typedef expression<T>                               expression_t;
      typedef symbol_table<T>                             symbol_table_t;
      typedef typename expression<T>::symtab_list_t       symbol_table_list_t;
      typedef details::vector_holder<T>*                  vector_holder_ptr;

      typedef typename details::functor_t<T> functor_t;
      typedef typename functor_t::qfunc_t    quaternary_functor_t;
      typedef typename functor_t::tfunc_t    trinary_functor_t;
      typedef typename functor_t::bfunc_t    binary_functor_t;
      typedef typename functor_t::ufunc_t    unary_functor_t;

      typedef details::operator_type operator_t;

      typedef std::map<operator_t, unary_functor_t  > unary_op_map_t;
      typedef std::map<operator_t, binary_functor_t > binary_op_map_t;
      typedef std::map<operator_t, trinary_functor_t> trinary_op_map_t;

      typedef std::map<std::string,std::pair<trinary_functor_t   ,operator_t> > sf3_map_t;
      typedef std::map<std::string,std::pair<quaternary_functor_t,operator_t> > sf4_map_t;

      typedef std::map<binary_functor_t,operator_t> inv_binary_op_map_t;
      typedef std::multimap<std::string,details::base_operation_t,details::ilesscompare> base_ops_map_t;
      typedef std::set<std::string,details::ilesscompare> disabled_func_set_t;

      typedef details::T0oT1_define<T, cref_t , cref_t > vov_t;
      typedef details::T0oT1_define<T, const_t, cref_t > cov_t;
      typedef details::T0oT1_define<T, cref_t , const_t> voc_t;

      typedef details::T0oT1oT2_define<T, cref_t , cref_t , cref_t > vovov_t;
      typedef details::T0oT1oT2_define<T, cref_t , cref_t , const_t> vovoc_t;
      typedef details::T0oT1oT2_define<T, cref_t , const_t, cref_t > vocov_t;
      typedef details::T0oT1oT2_define<T, const_t, cref_t , cref_t > covov_t;
      typedef details::T0oT1oT2_define<T, const_t, cref_t , const_t> covoc_t;
      typedef details::T0oT1oT2_define<T, const_t, const_t, cref_t > cocov_t;
      typedef details::T0oT1oT2_define<T, cref_t , const_t, const_t> vococ_t;

      typedef details::T0oT1oT2oT3_define<T, cref_t , cref_t , cref_t , cref_t > vovovov_t;
      typedef details::T0oT1oT2oT3_define<T, cref_t , cref_t , cref_t , const_t> vovovoc_t;
      typedef details::T0oT1oT2oT3_define<T, cref_t , cref_t , const_t, cref_t > vovocov_t;
      typedef details::T0oT1oT2oT3_define<T, cref_t , const_t, cref_t , cref_t > vocovov_t;
      typedef details::T0oT1oT2oT3_define<T, const_t, cref_t , cref_t , cref_t > covovov_t;

      typedef details::T0oT1oT2oT3_define<T, const_t, cref_t , const_t, cref_t > covocov_t;
      typedef details::T0oT1oT2oT3_define<T, cref_t , const_t, cref_t , const_t> vocovoc_t;
      typedef details::T0oT1oT2oT3_define<T, const_t, cref_t , cref_t , const_t> covovoc_t;
      typedef details::T0oT1oT2oT3_define<T, cref_t , const_t, const_t, cref_t > vococov_t;

      typedef results_context<T> results_context_t;

      typedef parser_helper prsrhlpr_t;

      struct scope_element
      {
         enum element_type
         {
            e_none    ,
            e_variable,
            e_vector  ,
            e_vecelem ,
            e_string
         };

         typedef details::vector_holder<T> vector_holder_t;
         typedef variable_node_t*          variable_node_ptr;
         typedef vector_holder_t*          vector_holder_ptr;
         typedef expression_node_t*        expression_node_ptr;

         scope_element()
         : name("???")
         , size (std::numeric_limits<std::size_t>::max())
         , index(std::numeric_limits<std::size_t>::max())
         , depth(std::numeric_limits<std::size_t>::max())
         , ref_count(0)
         , ip_index (0)
         , type (e_none)
         , active(false)
         , data     (0)
         , var_node (0)
         , vec_node (0)
         {}

         bool operator < (const scope_element& se) const
         {
            if (ip_index < se.ip_index)
               return true;
            else if (ip_index > se.ip_index)
               return false;
            else if (depth < se.depth)
               return true;
            else if (depth > se.depth)
               return false;
            else if (index < se.index)
               return true;
            else if (index > se.index)
               return false;
            else
               return (name < se.name);
         }

         void clear()
         {
            name   = "???";
            size   = std::numeric_limits<std::size_t>::max();
            index  = std::numeric_limits<std::size_t>::max();
            depth  = std::numeric_limits<std::size_t>::max();
            type   = e_none;
            active = false;
            ref_count = 0;
            ip_index  = 0;
            data      = 0;
            var_node  = 0;
            vec_node  = 0;
         }

         std::string  name;
         std::size_t  size;
         std::size_t  index;
         std::size_t  depth;
         std::size_t  ref_count;
         std::size_t  ip_index;
         element_type type;
         bool         active;
         void*        data;
         expression_node_ptr var_node;
         vector_holder_ptr   vec_node;
      };

      class scope_element_manager
      {
      public:

         typedef expression_node_t* expression_node_ptr;
         typedef variable_node_t*   variable_node_ptr;
         typedef parser<T>          parser_t;

         explicit scope_element_manager(parser<T>& p)
         : parser_(p)
         , input_param_cnt_(0)
         {}

         inline std::size_t size() const
         {
            return element_.size();
         }

         inline bool empty() const
         {
            return element_.empty();
         }

         inline scope_element& get_element(const std::size_t& index)
         {
            if (index < element_.size())
               return element_[index];
            else
               return null_element_;
         }

         inline scope_element& get_element(const std::string& var_name,
                                           const std::size_t index = std::numeric_limits<std::size_t>::max())
         {
            const std::size_t current_depth = parser_.state_.scope_depth;

            for (std::size_t i = 0; i < element_.size(); ++i)
            {
               scope_element& se = element_[i];

               if (se.depth > current_depth)
                  continue;
               else if (
                         details::imatch(se.name, var_name) &&
                         (se.index == index)
                       )
                  return se;
            }

            return null_element_;
         }

         inline scope_element& get_active_element(const std::string& var_name,
                                                  const std::size_t index = std::numeric_limits<std::size_t>::max())
         {
            const std::size_t current_depth = parser_.state_.scope_depth;

            for (std::size_t i = 0; i < element_.size(); ++i)
            {
               scope_element& se = element_[i];

               if (se.depth > current_depth)
                  continue;
               else if (
                         details::imatch(se.name, var_name) &&
                         (se.index == index)                &&
                         (se.active)
                       )
                  return se;
            }

            return null_element_;
         }

         inline bool add_element(const scope_element& se)
         {
            for (std::size_t i = 0; i < element_.size(); ++i)
            {
               scope_element& cse = element_[i];

               if (
                    details::imatch(cse.name, se.name) &&
                    (cse.depth <= se.depth)            &&
                    (cse.index == se.index)            &&
                    (cse.size  == se.size )            &&
                    (cse.type  == se.type )            &&
                    (cse.active)
                  )
                  return false;
            }

            element_.push_back(se);
            std::sort(element_.begin(),element_.end());

            return true;
         }

         inline void deactivate(const std::size_t& scope_depth)
         {
            exprtk_debug(("deactivate() - Scope depth: %d\n",
                          static_cast<int>(parser_.state_.scope_depth)));

            for (std::size_t i = 0; i < element_.size(); ++i)
            {
               scope_element& se = element_[i];

               if (se.active && (se.depth >= scope_depth))
               {
                  exprtk_debug(("deactivate() - element[%02d] '%s'\n",
                                static_cast<int>(i),
                                se.name.c_str()));

                  se.active = false;
               }
            }
         }

         inline void free_element(scope_element& se)
         {
            exprtk_debug(("free_element() - se[%s]\n", se.name.c_str()));

            switch (se.type)
            {
               case scope_element::e_variable   : delete reinterpret_cast<T*>(se.data);
                                                  delete se.var_node;
                                                  break;

               case scope_element::e_vector     : delete[] reinterpret_cast<T*>(se.data);
                                                  delete se.vec_node;
                                                  break;

               case scope_element::e_vecelem    : delete se.var_node;
                                                  break;

               default                          : return;
            }

            se.clear();
         }

         inline void cleanup()
         {
            for (std::size_t i = 0; i < element_.size(); ++i)
            {
               free_element(element_[i]);
            }

            element_.clear();

            input_param_cnt_ = 0;
         }

         inline std::size_t next_ip_index()
         {
            return ++input_param_cnt_;
         }

         inline expression_node_ptr get_variable(const T& v)
         {
            for (std::size_t i = 0; i < element_.size(); ++i)
            {
               scope_element& se = element_[i];

               if (
                    se.active   &&
                    se.var_node &&
                    details::is_variable_node(se.var_node)
                  )
               {
                  variable_node_ptr vn = reinterpret_cast<variable_node_ptr>(se.var_node);

                  if (&(vn->ref()) == (&v))
                  {
                     return se.var_node;
                  }
               }
            }

            return expression_node_ptr(0);
         }

      private:

         scope_element_manager(const scope_element_manager&) exprtk_delete;
         scope_element_manager& operator=(const scope_element_manager&) exprtk_delete;

         parser_t& parser_;
         std::vector<scope_element> element_;
         scope_element null_element_;
         std::size_t input_param_cnt_;
      };

      class scope_handler
      {
      public:

         typedef parser<T> parser_t;

         explicit scope_handler(parser<T>& p)
         : parser_(p)
         {
            parser_.state_.scope_depth++;
            #ifdef exprtk_enable_debugging
            const std::string depth(2 * parser_.state_.scope_depth,'-');
            exprtk_debug(("%s> Scope Depth: %02d\n",
                          depth.c_str(),
                          static_cast<int>(parser_.state_.scope_depth)));
            #endif
         }

        ~scope_handler()
         {
            parser_.sem_.deactivate(parser_.state_.scope_depth);
            parser_.state_.scope_depth--;
            #ifdef exprtk_enable_debugging
            const std::string depth(2 * parser_.state_.scope_depth,'-');
            exprtk_debug(("<%s Scope Depth: %02d\n",
                          depth.c_str(),
                          static_cast<int>(parser_.state_.scope_depth)));
            #endif
         }

      private:

         scope_handler(const scope_handler&) exprtk_delete;
         scope_handler& operator=(const scope_handler&) exprtk_delete;

         parser_t& parser_;
      };

      class stack_limit_handler
      {
      public:

         typedef parser<T> parser_t;

         explicit stack_limit_handler(parser<T>& p)
         : parser_(p)
         , limit_exceeded_(false)
         {
            if (++parser_.state_.stack_depth > parser_.settings_.max_stack_depth_)
            {
               limit_exceeded_ = true;
               parser_.set_error(
                  make_error(parser_error::e_parser,
                     "ERR000 - Current stack depth " + details::to_str(parser_.state_.stack_depth) +
                     " exceeds maximum allowed stack depth of " + details::to_str(parser_.settings_.max_stack_depth_),
                     exprtk_error_location));
            }
         }

        ~stack_limit_handler()
         {
            parser_.state_.stack_depth--;
         }

         bool operator!()
         {
            return limit_exceeded_;
         }

      private:

         stack_limit_handler(const stack_limit_handler&) exprtk_delete;
         stack_limit_handler& operator=(const stack_limit_handler&) exprtk_delete;

         parser_t& parser_;
         bool limit_exceeded_;
      };

      struct symtab_store
      {
         symbol_table_list_t symtab_list_;

         typedef typename symbol_table_t::local_data_t local_data_t;
         typedef typename symbol_table_t::variable_ptr variable_ptr;
         typedef typename symbol_table_t::function_ptr function_ptr;
         typedef typename symbol_table_t::vector_holder_ptr    vector_holder_ptr;
         typedef typename symbol_table_t::vararg_function_ptr  vararg_function_ptr;
         typedef typename symbol_table_t::generic_function_ptr generic_function_ptr;

         inline bool empty() const
         {
            return symtab_list_.empty();
         }

         inline void clear()
         {
            symtab_list_.clear();
         }

         inline bool valid() const
         {
            if (!empty())
            {
               for (std::size_t i = 0; i < symtab_list_.size(); ++i)
               {
                  if (symtab_list_[i].valid())
                     return true;
               }
            }

            return false;
         }

         inline bool valid_symbol(const std::string& symbol) const
         {
            if (!symtab_list_.empty())
               return symtab_list_[0].valid_symbol(symbol);
            else
               return false;
         }

         inline bool valid_function_name(const std::string& symbol) const
         {
            if (!symtab_list_.empty())
               return symtab_list_[0].valid_function(symbol);
            else
               return false;
         }

         inline variable_ptr get_variable(const std::string& variable_name) const
         {
            if (!valid_symbol(variable_name))
               return reinterpret_cast<variable_ptr>(0);

            variable_ptr result = reinterpret_cast<variable_ptr>(0);

            for (std::size_t i = 0; i < symtab_list_.size(); ++i)
            {
               if (!symtab_list_[i].valid())
                  continue;
               else
                  result = local_data(i)
                              .variable_store.get(variable_name);

               if (result) break;
            }

            return result;
         }

         inline variable_ptr get_variable(const T& var_ref) const
         {
            variable_ptr result = reinterpret_cast<variable_ptr>(0);

            for (std::size_t i = 0; i < symtab_list_.size(); ++i)
            {
               if (!symtab_list_[i].valid())
                  continue;
               else
                  result = local_data(i).variable_store
                              .get_from_varptr(reinterpret_cast<const void*>(&var_ref));

               if (result) break;
            }

            return result;
         }

         inline function_ptr get_function(const std::string& function_name) const
         {
            if (!valid_function_name(function_name))
               return reinterpret_cast<function_ptr>(0);

            function_ptr result = reinterpret_cast<function_ptr>(0);

            for (std::size_t i = 0; i < symtab_list_.size(); ++i)
            {
               if (!symtab_list_[i].valid())
                  continue;
               else
                  result = local_data(i)
                              .function_store.get(function_name);

               if (result) break;
            }

            return result;
         }

         inline vararg_function_ptr get_vararg_function(const std::string& vararg_function_name) const
         {
            if (!valid_function_name(vararg_function_name))
               return reinterpret_cast<vararg_function_ptr>(0);

            vararg_function_ptr result = reinterpret_cast<vararg_function_ptr>(0);

            for (std::size_t i = 0; i < symtab_list_.size(); ++i)
            {
               if (!symtab_list_[i].valid())
                  continue;
               else
                  result = local_data(i)
                              .vararg_function_store.get(vararg_function_name);

               if (result) break;
            }

            return result;
         }

         inline generic_function_ptr get_generic_function(const std::string& function_name) const
         {
            if (!valid_function_name(function_name))
               return reinterpret_cast<generic_function_ptr>(0);

            generic_function_ptr result = reinterpret_cast<generic_function_ptr>(0);

            for (std::size_t i = 0; i < symtab_list_.size(); ++i)
            {
               if (!symtab_list_[i].valid())
                  continue;
               else
                  result = local_data(i)
                              .generic_function_store.get(function_name);

               if (result) break;
            }

            return result;
         }

         inline generic_function_ptr get_string_function(const std::string& function_name) const
         {
            if (!valid_function_name(function_name))
               return reinterpret_cast<generic_function_ptr>(0);

            generic_function_ptr result = reinterpret_cast<generic_function_ptr>(0);

            for (std::size_t i = 0; i < symtab_list_.size(); ++i)
            {
               if (!symtab_list_[i].valid())
                  continue;
               else
                  result =
                     local_data(i).string_function_store.get(function_name);

               if (result) break;
            }

            return result;
         }

         inline generic_function_ptr get_overload_function(const std::string& function_name) const
         {
            if (!valid_function_name(function_name))
               return reinterpret_cast<generic_function_ptr>(0);

            generic_function_ptr result = reinterpret_cast<generic_function_ptr>(0);

            for (std::size_t i = 0; i < symtab_list_.size(); ++i)
            {
               if (!symtab_list_[i].valid())
                  continue;
               else
                  result =
                     local_data(i).overload_function_store.get(function_name);

               if (result) break;
            }

            return result;
         }

         inline vector_holder_ptr get_vector(const std::string& vector_name) const
         {
            if (!valid_symbol(vector_name))
               return reinterpret_cast<vector_holder_ptr>(0);

            vector_holder_ptr result = reinterpret_cast<vector_holder_ptr>(0);

            for (std::size_t i = 0; i < symtab_list_.size(); ++i)
            {
               if (!symtab_list_[i].valid())
                  continue;
               else
                  result =
                     local_data(i).vector_store.get(vector_name);

               if (result) break;
            }

            return result;
         }

         inline bool is_constant_node(const std::string& symbol_name) const
         {
            if (!valid_symbol(symbol_name))
               return false;

            for (std::size_t i = 0; i < symtab_list_.size(); ++i)
            {
               if (!symtab_list_[i].valid())
                  continue;
               else if (local_data(i).variable_store.is_constant(symbol_name))
                  return true;
            }

            return false;
         }

         inline bool symbol_exists(const std::string& symbol) const
         {
            for (std::size_t i = 0; i < symtab_list_.size(); ++i)
            {
               if (!symtab_list_[i].valid())
                  continue;
               else if (symtab_list_[i].symbol_exists(symbol))
                  return true;
            }

            return false;
         }

         inline bool is_variable(const std::string& variable_name) const
         {
            for (std::size_t i = 0; i < symtab_list_.size(); ++i)
            {
               if (!symtab_list_[i].valid())
                  continue;
               else if (
                         symtab_list_[i].local_data().variable_store
                           .symbol_exists(variable_name)
                       )
                  return true;
            }

            return false;
         }

         inline bool is_function(const std::string& function_name) const
         {
            for (std::size_t i = 0; i < symtab_list_.size(); ++i)
            {
               if (!symtab_list_[i].valid())
                  continue;
               else if (
                         local_data(i).vararg_function_store
                           .symbol_exists(function_name)
                       )
                  return true;
            }

            return false;
         }

         inline bool is_vararg_function(const std::string& vararg_function_name) const
         {
            for (std::size_t i = 0; i < symtab_list_.size(); ++i)
            {
               if (!symtab_list_[i].valid())
                  continue;
               else if (
                         local_data(i).vararg_function_store
                           .symbol_exists(vararg_function_name)
                       )
                  return true;
            }

            return false;
         }

         inline bool is_vector(const std::string& vector_name) const
         {
            for (std::size_t i = 0; i < symtab_list_.size(); ++i)
            {
               if (!symtab_list_[i].valid())
                  continue;
               else if (
                         local_data(i).vector_store
                           .symbol_exists(vector_name)
                       )
                  return true;
            }

            return false;
         }

         inline std::string get_variable_name(const expression_node_ptr& ptr) const
         {
            return local_data().variable_store.entity_name(ptr);
         }

         inline std::string get_vector_name(const vector_holder_ptr& ptr) const
         {
            return local_data().vector_store.entity_name(ptr);
         }

         inline local_data_t& local_data(const std::size_t& index = 0)
         {
            return symtab_list_[index].local_data();
         }

         inline const local_data_t& local_data(const std::size_t& index = 0) const
         {
            return symtab_list_[index].local_data();
         }

         inline symbol_table_t& get_symbol_table(const std::size_t& index = 0)
         {
            return symtab_list_[index];
         }
      };

      struct parser_state
      {
         parser_state()
         : type_check_enabled(true)
         {
            reset();
         }

         void reset()
         {
            parsing_return_stmt     = false;
            parsing_break_stmt      = false;
            return_stmt_present     = false;
            side_effect_present     = false;
            scope_depth             = 0;
            stack_depth             = 0;
            parsing_loop_stmt_count = 0;
         }

         #ifndef exprtk_enable_debugging
         void activate_side_effect(const std::string&)
         #else
         void activate_side_effect(const std::string& source)
         #endif
         {
            if (!side_effect_present)
            {
               side_effect_present = true;

               exprtk_debug(("activate_side_effect() - caller: %s\n",source.c_str()));
            }
         }

         bool parsing_return_stmt;
         bool parsing_break_stmt;
         bool return_stmt_present;
         bool side_effect_present;
         bool type_check_enabled;
         std::size_t scope_depth;
         std::size_t stack_depth;
         std::size_t parsing_loop_stmt_count;
      };

   public:

      struct unknown_symbol_resolver
      {

         enum usr_symbol_type
         {
            e_usr_unknown_type  = 0,
            e_usr_variable_type = 1,
            e_usr_constant_type = 2
         };

         enum usr_mode
         {
            e_usrmode_default  = 0,
            e_usrmode_extended = 1
         };

         usr_mode mode;

         unknown_symbol_resolver(const usr_mode m = e_usrmode_default)
         : mode(m)
         {}

         virtual ~unknown_symbol_resolver() {}

         virtual bool process(const std::string& /*unknown_symbol*/,
                              usr_symbol_type&   st,
                              T&                 default_value,
                              std::string&       error_message)
         {
            if (e_usrmode_default != mode)
               return false;

            st = e_usr_variable_type;
            default_value = T(0);
            error_message.clear();

            return true;
         }

         virtual bool process(const std::string& /* unknown_symbol */,
                              symbol_table_t&    /* symbol_table   */,
                              std::string&       /* error_message  */)
         {
            return false;
         }
      };

      enum collect_type
      {
         e_ct_none        = 0,
         e_ct_variables   = 1,
         e_ct_functions   = 2,
         e_ct_assignments = 4
      };

      enum symbol_type
      {
         e_st_unknown        = 0,
         e_st_variable       = 1,
         e_st_vector         = 2,
         e_st_vecelem        = 3,
         e_st_string         = 4,
         e_st_function       = 5,
         e_st_local_variable = 6,
         e_st_local_vector   = 7,
         e_st_local_string   = 8
      };

      class dependent_entity_collector
      {
      public:

         typedef std::pair<std::string,symbol_type> symbol_t;
         typedef std::vector<symbol_t> symbol_list_t;

         dependent_entity_collector(const std::size_t options = e_ct_none)
         : options_(options)
         , collect_variables_  ((options_ & e_ct_variables  ) == e_ct_variables  )
         , collect_functions_  ((options_ & e_ct_functions  ) == e_ct_functions  )
         , collect_assignments_((options_ & e_ct_assignments) == e_ct_assignments)
         , return_present_   (false)
         , final_stmt_return_(false)
         {}

         template <typename Allocator,
                   template <typename, typename> class Sequence>
         inline std::size_t symbols(Sequence<symbol_t,Allocator>& symbols_list)
         {
            if (!collect_variables_ && !collect_functions_)
               return 0;
            else if (symbol_name_list_.empty())
               return 0;

            for (std::size_t i = 0; i < symbol_name_list_.size(); ++i)
            {
               details::case_normalise(symbol_name_list_[i].first);
            }

            std::sort(symbol_name_list_.begin(),symbol_name_list_.end());

            std::unique_copy(symbol_name_list_.begin(),
                             symbol_name_list_.end  (),
                             std::back_inserter(symbols_list));

            return symbols_list.size();
         }

         template <typename Allocator,
                   template <typename, typename> class Sequence>
         inline std::size_t assignment_symbols(Sequence<symbol_t,Allocator>& assignment_list)
         {
            if (!collect_assignments_)
               return 0;
            else if (assignment_name_list_.empty())
               return 0;

            for (std::size_t i = 0; i < assignment_name_list_.size(); ++i)
            {
               details::case_normalise(assignment_name_list_[i].first);
            }

            std::sort(assignment_name_list_.begin(),assignment_name_list_.end());

            std::unique_copy(assignment_name_list_.begin(),
                             assignment_name_list_.end  (),
                             std::back_inserter(assignment_list));

            return assignment_list.size();
         }

         void clear()
         {
            symbol_name_list_    .clear();
            assignment_name_list_.clear();
            retparam_list_       .clear();
            return_present_    = false;
            final_stmt_return_ = false;
         }

         bool& collect_variables()
         {
            return collect_variables_;
         }

         bool& collect_functions()
         {
            return collect_functions_;
         }

         bool& collect_assignments()
         {
            return collect_assignments_;
         }

         bool return_present() const
         {
            return return_present_;
         }

         bool final_stmt_return() const
         {
            return final_stmt_return_;
         }

         typedef std::vector<std::string> retparam_list_t;

         retparam_list_t return_param_type_list() const
         {
            return retparam_list_;
         }

      private:

         inline void add_symbol(const std::string& symbol, const symbol_type st)
         {
            switch (st)
            {
               case e_st_variable       :
               case e_st_vector         :
               case e_st_string         :
               case e_st_local_variable :
               case e_st_local_vector   :
               case e_st_local_string   : if (collect_variables_)
                                             symbol_name_list_
                                                .push_back(std::make_pair(symbol, st));
                                          break;

               case e_st_function       : if (collect_functions_)
                                             symbol_name_list_
                                                .push_back(std::make_pair(symbol, st));
                                          break;

               default                  : return;
            }
         }

         inline void add_assignment(const std::string& symbol, const symbol_type st)
         {
            switch (st)
            {
               case e_st_variable       :
               case e_st_vector         :
               case e_st_string         : if (collect_assignments_)
                                             assignment_name_list_
                                                .push_back(std::make_pair(symbol, st));
                                          break;

               default                  : return;
            }
         }

         std::size_t options_;
         bool collect_variables_;
         bool collect_functions_;
         bool collect_assignments_;
         bool return_present_;
         bool final_stmt_return_;
         symbol_list_t symbol_name_list_;
         symbol_list_t assignment_name_list_;
         retparam_list_t retparam_list_;

         friend class parser<T>;
      };

      class settings_store
      {
      private:

         typedef std::set<std::string,details::ilesscompare> disabled_entity_set_t;
         typedef disabled_entity_set_t::iterator des_itr_t;

      public:

         enum settings_compilation_options
         {
            e_unknown              =    0,
            e_replacer             =    1,
            e_joiner               =    2,
            e_numeric_check        =    4,
            e_bracket_check        =    8,
            e_sequence_check       =   16,
            e_commutative_check    =   32,
            e_strength_reduction   =   64,
            e_disable_vardef       =  128,
            e_collect_vars         =  256,
            e_collect_funcs        =  512,
            e_collect_assings      = 1024,
            e_disable_usr_on_rsrvd = 2048,
            e_disable_zero_return  = 4096
         };

         enum settings_base_funcs
         {
            e_bf_unknown = 0,
            e_bf_abs       , e_bf_acos     , e_bf_acosh    , e_bf_asin    ,
            e_bf_asinh     , e_bf_atan     , e_bf_atan2    , e_bf_atanh   ,
            e_bf_avg       , e_bf_ceil     , e_bf_clamp    , e_bf_cos     ,
            e_bf_cosh      , e_bf_cot      , e_bf_csc      , e_bf_equal   ,
            e_bf_erf       , e_bf_erfc     , e_bf_exp      , e_bf_expm1   ,
            e_bf_floor     , e_bf_frac     , e_bf_hypot    , e_bf_iclamp  ,
            e_bf_like      , e_bf_log      , e_bf_log10    , e_bf_log1p   ,
            e_bf_log2      , e_bf_logn     , e_bf_mand     , e_bf_max     ,
            e_bf_min       , e_bf_mod      , e_bf_mor      , e_bf_mul     ,
            e_bf_ncdf      , e_bf_pow      , e_bf_root     , e_bf_round   ,
            e_bf_roundn    , e_bf_sec      , e_bf_sgn      , e_bf_sin     ,
            e_bf_sinc      , e_bf_sinh     , e_bf_sqrt     , e_bf_sum     ,
            e_bf_swap      , e_bf_tan      , e_bf_tanh     , e_bf_trunc   ,
            e_bf_not_equal , e_bf_inrange  , e_bf_deg2grad , e_bf_deg2rad ,
            e_bf_rad2deg   , e_bf_grad2deg
         };

         enum settings_control_structs
         {
            e_ctrl_unknown = 0,
            e_ctrl_ifelse,
            e_ctrl_switch,
            e_ctrl_for_loop,
            e_ctrl_while_loop,
            e_ctrl_repeat_loop,
            e_ctrl_return
         };

         enum settings_logic_opr
         {
            e_logic_unknown = 0,
            e_logic_and, e_logic_nand , e_logic_nor ,
            e_logic_not, e_logic_or   , e_logic_xnor,
            e_logic_xor, e_logic_scand, e_logic_scor
         };

         enum settings_arithmetic_opr
         {
            e_arith_unknown = 0,
            e_arith_add, e_arith_sub, e_arith_mul,
            e_arith_div, e_arith_mod, e_arith_pow
         };

         enum settings_assignment_opr
         {
            e_assign_unknown = 0,
            e_assign_assign, e_assign_addass, e_assign_subass,
            e_assign_mulass, e_assign_divass, e_assign_modass
         };

         enum settings_inequality_opr
         {
            e_ineq_unknown = 0,
            e_ineq_lt   , e_ineq_lte, e_ineq_eq    ,
            e_ineq_equal, e_ineq_ne , e_ineq_nequal,
            e_ineq_gte  , e_ineq_gt
         };

         static const std::size_t compile_all_opts =
                                     e_replacer          +
                                     e_joiner            +
                                     e_numeric_check     +
                                     e_bracket_check     +
                                     e_sequence_check    +
                                     e_commutative_check +
                                     e_strength_reduction;

         settings_store(const std::size_t compile_options = compile_all_opts)
         : max_stack_depth_(400)
         , max_node_depth_(10000)
         {
           load_compile_options(compile_options);
         }

         settings_store& enable_all_base_functions()
         {
            disabled_func_set_.clear();
            return (*this);
         }

         settings_store& enable_all_control_structures()
         {
            disabled_ctrl_set_.clear();
            return (*this);
         }

         settings_store& enable_all_logic_ops()
         {
            disabled_logic_set_.clear();
            return (*this);
         }

         settings_store& enable_all_arithmetic_ops()
         {
            disabled_arithmetic_set_.clear();
            return (*this);
         }

         settings_store& enable_all_assignment_ops()
         {
            disabled_assignment_set_.clear();
            return (*this);
         }

         settings_store& enable_all_inequality_ops()
         {
            disabled_inequality_set_.clear();
            return (*this);
         }

         settings_store& enable_local_vardef()
         {
            disable_vardef_ = false;
            return (*this);
         }

         settings_store& disable_all_base_functions()
         {
            std::copy(details::base_function_list,
                      details::base_function_list + details::base_function_list_size,
                      std::insert_iterator<disabled_entity_set_t>
                        (disabled_func_set_, disabled_func_set_.begin()));
            return (*this);
         }

         settings_store& disable_all_control_structures()
         {
            std::copy(details::cntrl_struct_list,
                      details::cntrl_struct_list + details::cntrl_struct_list_size,
                      std::insert_iterator<disabled_entity_set_t>
                        (disabled_ctrl_set_, disabled_ctrl_set_.begin()));
            return (*this);
         }

         settings_store& disable_all_logic_ops()
         {
            std::copy(details::logic_ops_list,
                      details::logic_ops_list + details::logic_ops_list_size,
                      std::insert_iterator<disabled_entity_set_t>
                        (disabled_logic_set_, disabled_logic_set_.begin()));
            return (*this);
         }

         settings_store& disable_all_arithmetic_ops()
         {
            std::copy(details::arithmetic_ops_list,
                      details::arithmetic_ops_list + details::arithmetic_ops_list_size,
                      std::insert_iterator<disabled_entity_set_t>
                        (disabled_arithmetic_set_, disabled_arithmetic_set_.begin()));
            return (*this);
         }

         settings_store& disable_all_assignment_ops()
         {
            std::copy(details::assignment_ops_list,
                      details::assignment_ops_list + details::assignment_ops_list_size,
                      std::insert_iterator<disabled_entity_set_t>
                        (disabled_assignment_set_, disabled_assignment_set_.begin()));
            return (*this);
         }

         settings_store& disable_all_inequality_ops()
         {
            std::copy(details::inequality_ops_list,
                      details::inequality_ops_list + details::inequality_ops_list_size,
                      std::insert_iterator<disabled_entity_set_t>
                        (disabled_inequality_set_, disabled_inequality_set_.begin()));
            return (*this);
         }

         settings_store& disable_local_vardef()
         {
            disable_vardef_ = true;
            return (*this);
         }

         bool replacer_enabled           () const { return enable_replacer_;           }
         bool commutative_check_enabled  () const { return enable_commutative_check_;  }
         bool joiner_enabled             () const { return enable_joiner_;             }
         bool numeric_check_enabled      () const { return enable_numeric_check_;      }
         bool bracket_check_enabled      () const { return enable_bracket_check_;      }
         bool sequence_check_enabled     () const { return enable_sequence_check_;     }
         bool strength_reduction_enabled () const { return enable_strength_reduction_; }
         bool collect_variables_enabled  () const { return enable_collect_vars_;       }
         bool collect_functions_enabled  () const { return enable_collect_funcs_;      }
         bool collect_assignments_enabled() const { return enable_collect_assings_;    }
         bool vardef_disabled            () const { return disable_vardef_;            }
         bool rsrvd_sym_usr_disabled     () const { return disable_rsrvd_sym_usr_;     }
         bool zero_return_disabled       () const { return disable_zero_return_;       }

         bool function_enabled(const std::string& function_name) const
         {
            if (disabled_func_set_.empty())
               return true;
            else
               return (disabled_func_set_.end() == disabled_func_set_.find(function_name));
         }

         bool control_struct_enabled(const std::string& control_struct) const
         {
            if (disabled_ctrl_set_.empty())
               return true;
            else
               return (disabled_ctrl_set_.end() == disabled_ctrl_set_.find(control_struct));
         }

         bool logic_enabled(const std::string& logic_operation) const
         {
            if (disabled_logic_set_.empty())
               return true;
            else
               return (disabled_logic_set_.end() == disabled_logic_set_.find(logic_operation));
         }

         bool arithmetic_enabled(const details::operator_type& arithmetic_operation) const
         {
            if (disabled_logic_set_.empty())
               return true;
            else
               return disabled_arithmetic_set_.end() == disabled_arithmetic_set_
                                                            .find(arith_opr_to_string(arithmetic_operation));
         }

         bool assignment_enabled(const details::operator_type& assignment) const
         {
            if (disabled_assignment_set_.empty())
               return true;
            else
               return disabled_assignment_set_.end() == disabled_assignment_set_
                                                           .find(assign_opr_to_string(assignment));
         }

         bool inequality_enabled(const details::operator_type& inequality) const
         {
            if (disabled_inequality_set_.empty())
               return true;
            else
               return disabled_inequality_set_.end() == disabled_inequality_set_
                                                           .find(inequality_opr_to_string(inequality));
         }

         bool function_disabled(const std::string& function_name) const
         {
            if (disabled_func_set_.empty())
               return false;
            else
               return (disabled_func_set_.end() != disabled_func_set_.find(function_name));
         }

         bool control_struct_disabled(const std::string& control_struct) const
         {
            if (disabled_ctrl_set_.empty())
               return false;
            else
               return (disabled_ctrl_set_.end() != disabled_ctrl_set_.find(control_struct));
         }

         bool logic_disabled(const std::string& logic_operation) const
         {
            if (disabled_logic_set_.empty())
               return false;
            else
               return (disabled_logic_set_.end() != disabled_logic_set_.find(logic_operation));
         }

         bool assignment_disabled(const details::operator_type assignment_operation) const
         {
            if (disabled_assignment_set_.empty())
               return false;
            else
               return disabled_assignment_set_.end() != disabled_assignment_set_
                                                           .find(assign_opr_to_string(assignment_operation));
         }

         bool logic_disabled(const details::operator_type logic_operation) const
         {
            if (disabled_logic_set_.empty())
               return false;
            else
               return disabled_logic_set_.end() != disabled_logic_set_
                                                           .find(logic_opr_to_string(logic_operation));
         }

         bool arithmetic_disabled(const details::operator_type arithmetic_operation) const
         {
            if (disabled_arithmetic_set_.empty())
               return false;
            else
               return disabled_arithmetic_set_.end() != disabled_arithmetic_set_
                                                           .find(arith_opr_to_string(arithmetic_operation));
         }

         bool inequality_disabled(const details::operator_type& inequality) const
         {
            if (disabled_inequality_set_.empty())
               return false;
            else
               return disabled_inequality_set_.end() != disabled_inequality_set_
                                                           .find(inequality_opr_to_string(inequality));
         }

         settings_store& disable_base_function(settings_base_funcs bf)
         {
            if (
                 (e_bf_unknown != bf) &&
                 (static_cast<std::size_t>(bf) < (details::base_function_list_size + 1))
               )
            {
               disabled_func_set_.insert(details::base_function_list[bf - 1]);
            }

            return (*this);
         }

         settings_store& disable_control_structure(settings_control_structs ctrl_struct)
         {
            if (
                 (e_ctrl_unknown != ctrl_struct) &&
                 (static_cast<std::size_t>(ctrl_struct) < (details::cntrl_struct_list_size + 1))
               )
            {
               disabled_ctrl_set_.insert(details::cntrl_struct_list[ctrl_struct - 1]);
            }

            return (*this);
         }

         settings_store& disable_logic_operation(settings_logic_opr logic)
         {
            if (
                 (e_logic_unknown != logic) &&
                 (static_cast<std::size_t>(logic) < (details::logic_ops_list_size + 1))
               )
            {
               disabled_logic_set_.insert(details::logic_ops_list[logic - 1]);
            }

            return (*this);
         }

         settings_store& disable_arithmetic_operation(settings_arithmetic_opr arithmetic)
         {
            if (
                 (e_arith_unknown != arithmetic) &&
                 (static_cast<std::size_t>(arithmetic) < (details::arithmetic_ops_list_size + 1))
               )
            {
               disabled_arithmetic_set_.insert(details::arithmetic_ops_list[arithmetic - 1]);
            }

            return (*this);
         }

         settings_store& disable_assignment_operation(settings_assignment_opr assignment)
         {
            if (
                 (e_assign_unknown != assignment) &&
                 (static_cast<std::size_t>(assignment) < (details::assignment_ops_list_size + 1))
               )
            {
               disabled_assignment_set_.insert(details::assignment_ops_list[assignment - 1]);
            }

            return (*this);
         }

         settings_store& disable_inequality_operation(settings_inequality_opr inequality)
         {
            if (
                 (e_ineq_unknown != inequality) &&
                 (static_cast<std::size_t>(inequality) < (details::inequality_ops_list_size + 1))
               )
            {
               disabled_inequality_set_.insert(details::inequality_ops_list[inequality - 1]);
            }

            return (*this);
         }

         settings_store& enable_base_function(settings_base_funcs bf)
         {
            if (
                 (e_bf_unknown != bf) &&
                 (static_cast<std::size_t>(bf) < (details::base_function_list_size + 1))
               )
            {
               const des_itr_t itr = disabled_func_set_.find(details::base_function_list[bf - 1]);

               if (disabled_func_set_.end() != itr)
               {
                  disabled_func_set_.erase(itr);
               }
            }

            return (*this);
         }

         settings_store& enable_control_structure(settings_control_structs ctrl_struct)
         {
            if (
                 (e_ctrl_unknown != ctrl_struct) &&
                 (static_cast<std::size_t>(ctrl_struct) < (details::cntrl_struct_list_size + 1))
               )
            {
               const des_itr_t itr = disabled_ctrl_set_.find(details::cntrl_struct_list[ctrl_struct - 1]);

               if (disabled_ctrl_set_.end() != itr)
               {
                  disabled_ctrl_set_.erase(itr);
               }
            }

            return (*this);
         }

         settings_store& enable_logic_operation(settings_logic_opr logic)
         {
            if (
                 (e_logic_unknown != logic) &&
                 (static_cast<std::size_t>(logic) < (details::logic_ops_list_size + 1))
               )
            {
               const des_itr_t itr = disabled_logic_set_.find(details::logic_ops_list[logic - 1]);

               if (disabled_logic_set_.end() != itr)
               {
                  disabled_logic_set_.erase(itr);
               }
            }

            return (*this);
         }

         settings_store& enable_arithmetic_operation(settings_arithmetic_opr arithmetic)
         {
            if (
                 (e_arith_unknown != arithmetic) &&
                 (static_cast<std::size_t>(arithmetic) < (details::arithmetic_ops_list_size + 1))
               )
            {
               const des_itr_t itr = disabled_arithmetic_set_.find(details::arithmetic_ops_list[arithmetic - 1]);

               if (disabled_arithmetic_set_.end() != itr)
               {
                  disabled_arithmetic_set_.erase(itr);
               }
            }

            return (*this);
         }

         settings_store& enable_assignment_operation(settings_assignment_opr assignment)
         {
            if (
                 (e_assign_unknown != assignment) &&
                 (static_cast<std::size_t>(assignment) < (details::assignment_ops_list_size + 1))
               )
            {
               const des_itr_t itr = disabled_assignment_set_.find(details::assignment_ops_list[assignment - 1]);

               if (disabled_assignment_set_.end() != itr)
               {
                  disabled_assignment_set_.erase(itr);
               }
            }

            return (*this);
         }

         settings_store& enable_inequality_operation(settings_inequality_opr inequality)
         {
            if (
                 (e_ineq_unknown != inequality) &&
                 (static_cast<std::size_t>(inequality) < (details::inequality_ops_list_size + 1))
               )
            {
               const des_itr_t itr = disabled_inequality_set_.find(details::inequality_ops_list[inequality - 1]);

               if (disabled_inequality_set_.end() != itr)
               {
                  disabled_inequality_set_.erase(itr);
               }
            }

            return (*this);
         }

         void set_max_stack_depth(const std::size_t max_stack_depth)
         {
            max_stack_depth_ = max_stack_depth;
         }

         void set_max_node_depth(const std::size_t max_node_depth)
         {
            max_node_depth_ = max_node_depth;
         }

      private:

         void load_compile_options(const std::size_t compile_options)
         {
            enable_replacer_           = (compile_options & e_replacer            ) == e_replacer;
            enable_joiner_             = (compile_options & e_joiner              ) == e_joiner;
            enable_numeric_check_      = (compile_options & e_numeric_check       ) == e_numeric_check;
            enable_bracket_check_      = (compile_options & e_bracket_check       ) == e_bracket_check;
            enable_sequence_check_     = (compile_options & e_sequence_check      ) == e_sequence_check;
            enable_commutative_check_  = (compile_options & e_commutative_check   ) == e_commutative_check;
            enable_strength_reduction_ = (compile_options & e_strength_reduction  ) == e_strength_reduction;
            enable_collect_vars_       = (compile_options & e_collect_vars        ) == e_collect_vars;
            enable_collect_funcs_      = (compile_options & e_collect_funcs       ) == e_collect_funcs;
            enable_collect_assings_    = (compile_options & e_collect_assings     ) == e_collect_assings;
            disable_vardef_            = (compile_options & e_disable_vardef      ) == e_disable_vardef;
            disable_rsrvd_sym_usr_     = (compile_options & e_disable_usr_on_rsrvd) == e_disable_usr_on_rsrvd;
            disable_zero_return_       = (compile_options & e_disable_zero_return ) == e_disable_zero_return;
         }

         std::string assign_opr_to_string(details::operator_type opr) const
         {
            switch (opr)
            {
               case details::e_assign : return ":=";
               case details::e_addass : return "+=";
               case details::e_subass : return "-=";
               case details::e_mulass : return "*=";
               case details::e_divass : return "/=";
               case details::e_modass : return "%=";
               default                : return ""  ;
            }
         }

         std::string arith_opr_to_string(details::operator_type opr) const
         {
            switch (opr)
            {
               case details::e_add : return "+";
               case details::e_sub : return "-";
               case details::e_mul : return "*";
               case details::e_div : return "/";
               case details::e_mod : return "%";
               default             : return "" ;
            }
         }

         std::string inequality_opr_to_string(details::operator_type opr) const
         {
            switch (opr)
            {
               case details::e_lt    : return "<" ;
               case details::e_lte   : return "<=";
               case details::e_eq    : return "==";
               case details::e_equal : return "=" ;
               case details::e_ne    : return "!=";
               case details::e_nequal: return "<>";
               case details::e_gte   : return ">=";
               case details::e_gt    : return ">" ;
               default               : return ""  ;
            }
         }

         std::string logic_opr_to_string(details::operator_type opr) const
         {
            switch (opr)
            {
               case details::e_and  : return "and" ;
               case details::e_or   : return "or"  ;
               case details::e_xor  : return "xor" ;
               case details::e_nand : return "nand";
               case details::e_nor  : return "nor" ;
               case details::e_xnor : return "xnor";
               case details::e_notl : return "not" ;
               default              : return ""    ;
            }
         }

         bool enable_replacer_;
         bool enable_joiner_;
         bool enable_numeric_check_;
         bool enable_bracket_check_;
         bool enable_sequence_check_;
         bool enable_commutative_check_;
         bool enable_strength_reduction_;
         bool enable_collect_vars_;
         bool enable_collect_funcs_;
         bool enable_collect_assings_;
         bool disable_vardef_;
         bool disable_rsrvd_sym_usr_;
         bool disable_zero_return_;

         disabled_entity_set_t disabled_func_set_ ;
         disabled_entity_set_t disabled_ctrl_set_ ;
         disabled_entity_set_t disabled_logic_set_;
         disabled_entity_set_t disabled_arithmetic_set_;
         disabled_entity_set_t disabled_assignment_set_;
         disabled_entity_set_t disabled_inequality_set_;

         std::size_t max_stack_depth_;
         std::size_t max_node_depth_;

         friend class parser<T>;
      };

      typedef settings_store settings_t;

      parser(const settings_t& settings = settings_t())
      : settings_(settings)
      , resolve_unknown_symbol_(false)
      , results_context_(0)
      , unknown_symbol_resolver_(reinterpret_cast<unknown_symbol_resolver*>(0))
        #ifdef _MSC_VER
        #pragma warning(push)
        #pragma warning (disable:4355)
        #endif
      , sem_(*this)
        #ifdef _MSC_VER
        #pragma warning(pop)
        #endif
      , operator_joiner_2_(2)
      , operator_joiner_3_(3)
      , loop_runtime_check_(0)
      {
         init_precompilation();

         load_operations_map           (base_ops_map_     );
         load_unary_operations_map     (unary_op_map_     );
         load_binary_operations_map    (binary_op_map_    );
         load_inv_binary_operations_map(inv_binary_op_map_);
         load_sf3_map                  (sf3_map_          );
         load_sf4_map                  (sf4_map_          );

         expression_generator_.init_synthesize_map();
         expression_generator_.set_parser(*this);
         expression_generator_.set_uom(unary_op_map_);
         expression_generator_.set_bom(binary_op_map_);
         expression_generator_.set_ibom(inv_binary_op_map_);
         expression_generator_.set_sf3m(sf3_map_);
         expression_generator_.set_sf4m(sf4_map_);
         expression_generator_.set_strength_reduction_state(settings_.strength_reduction_enabled());
      }

     ~parser() {}

      inline void init_precompilation()
      {
         dec_.collect_variables() =
            settings_.collect_variables_enabled();

         dec_.collect_functions() =
            settings_.collect_functions_enabled();

         dec_.collect_assignments() =
            settings_.collect_assignments_enabled();

         if (settings_.replacer_enabled())
         {
            symbol_replacer_.clear();
            symbol_replacer_.add_replace("true" , "1", lexer::token::e_number);
            symbol_replacer_.add_replace("false", "0", lexer::token::e_number);
            helper_assembly_.token_modifier_list.clear();
            helper_assembly_.register_modifier(&symbol_replacer_);
         }

         if (settings_.commutative_check_enabled())
         {
            for (std::size_t i = 0; i < details::reserved_words_size; ++i)
            {
               commutative_inserter_.ignore_symbol(details::reserved_words[i]);
            }

            helper_assembly_.token_inserter_list.clear();
            helper_assembly_.register_inserter(&commutative_inserter_);
         }

         if (settings_.joiner_enabled())
         {
            helper_assembly_.token_joiner_list.clear();
            helper_assembly_.register_joiner(&operator_joiner_2_);
            helper_assembly_.register_joiner(&operator_joiner_3_);
         }

         if (
              settings_.numeric_check_enabled () ||
              settings_.bracket_check_enabled () ||
              settings_.sequence_check_enabled()
            )
         {
            helper_assembly_.token_scanner_list.clear();

            if (settings_.numeric_check_enabled())
            {
               helper_assembly_.register_scanner(&numeric_checker_);
            }

            if (settings_.bracket_check_enabled())
            {
               helper_assembly_.register_scanner(&bracket_checker_);
            }

            if (settings_.sequence_check_enabled())
            {
               helper_assembly_.register_scanner(&sequence_validator_      );
               helper_assembly_.register_scanner(&sequence_validator_3tkns_);
            }
         }
      }

      inline bool compile(const std::string& expression_string, expression<T>& expr)
      {
         state_          .reset();
         error_list_     .clear();
         brkcnt_list_    .clear();
         synthesis_error_.clear();
         sem_            .cleanup();

         return_cleanup();

         expression_generator_.set_allocator(node_allocator_);

         if (expression_string.empty())
         {
            set_error(
               make_error(parser_error::e_syntax,
                          "ERR001 - Empty expression!",
                          exprtk_error_location));

            return false;
         }

         if (!init(expression_string))
         {
            process_lexer_errors();
            return false;
         }

         if (lexer().empty())
         {
            set_error(
               make_error(parser_error::e_syntax,
                          "ERR002 - Empty expression!",
                          exprtk_error_location));

            return false;
         }

         if (!run_assemblies())
         {
            return false;
         }

         symtab_store_.symtab_list_ = expr.get_symbol_table_list();
         dec_.clear();

         lexer().begin();

         next_token();

         expression_node_ptr e = parse_corpus();

         if ((0 != e) && (token_t::e_eof == current_token().type))
         {
            bool* retinvk_ptr = 0;

            if (state_.return_stmt_present)
            {
               dec_.return_present_ = true;

               e = expression_generator_
                     .return_envelope(e, results_context_, retinvk_ptr);
            }

            expr.set_expression(e);
            expr.set_retinvk(retinvk_ptr);

            register_local_vars(expr);
            register_return_results(expr);

            return !(!expr);
         }
         else
         {
            if (error_list_.empty())
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR003 - Invalid expression encountered",
                             exprtk_error_location));
            }

            if ((0 != e) && branch_deletable(e))
            {
               destroy_node(e);
            }

            dec_.clear    ();
            sem_.cleanup  ();
            return_cleanup();

            return false;
         }
      }

      inline expression_t compile(const std::string& expression_string, symbol_table_t& symtab)
      {
         expression_t expression;
         expression.register_symbol_table(symtab);
         compile(expression_string,expression);
         return expression;
      }

      void process_lexer_errors()
      {
         for (std::size_t i = 0; i < lexer().size(); ++i)
         {
            if (lexer()[i].is_error())
            {
               std::string diagnostic = "ERR004 - ";

               switch (lexer()[i].type)
               {
                  case lexer::token::e_error      : diagnostic += "General token error";
                                                    break;

                  case lexer::token::e_err_symbol : diagnostic += "Symbol error";
                                                    break;

                  case lexer::token::e_err_number : diagnostic += "Invalid numeric token";
                                                    break;

                  case lexer::token::e_err_string : diagnostic += "Invalid string token";
                                                    break;

                  case lexer::token::e_err_sfunc  : diagnostic += "Invalid special function token";
                                                    break;

                  default                         : diagnostic += "Unknown compiler error";
               }

               set_error(
                  make_error(parser_error::e_lexer,
                             lexer()[i],
                             diagnostic + ": " + lexer()[i].value,
                             exprtk_error_location));
            }
         }
      }

      inline bool run_assemblies()
      {
         if (settings_.commutative_check_enabled())
         {
            helper_assembly_.run_inserters(lexer());
         }

         if (settings_.joiner_enabled())
         {
            helper_assembly_.run_joiners(lexer());
         }

         if (settings_.replacer_enabled())
         {
            helper_assembly_.run_modifiers(lexer());
         }

         if (
              settings_.numeric_check_enabled () ||
              settings_.bracket_check_enabled () ||
              settings_.sequence_check_enabled()
            )
         {
            if (!helper_assembly_.run_scanners(lexer()))
            {
               if (helper_assembly_.error_token_scanner)
               {
                  lexer::helper::bracket_checker*            bracket_checker_ptr     = 0;
                  lexer::helper::numeric_checker*            numeric_checker_ptr     = 0;
                  lexer::helper::sequence_validator*         sequence_validator_ptr  = 0;
                  lexer::helper::sequence_validator_3tokens* sequence_validator3_ptr = 0;

                  if (0 != (bracket_checker_ptr = dynamic_cast<lexer::helper::bracket_checker*>(helper_assembly_.error_token_scanner)))
                  {
                     set_error(
                        make_error(parser_error::e_token,
                                   bracket_checker_ptr->error_token(),
                                   "ERR005 - Mismatched brackets: '" + bracket_checker_ptr->error_token().value + "'",
                                   exprtk_error_location));
                  }
                  else if (0 != (numeric_checker_ptr = dynamic_cast<lexer::helper::numeric_checker*>(helper_assembly_.error_token_scanner)))
                  {
                     for (std::size_t i = 0; i < numeric_checker_ptr->error_count(); ++i)
                     {
                        lexer::token error_token = lexer()[numeric_checker_ptr->error_index(i)];

                        set_error(
                           make_error(parser_error::e_token,
                                      error_token,
                                      "ERR006 - Invalid numeric token: '" + error_token.value + "'",
                                      exprtk_error_location));
                     }

                     if (numeric_checker_ptr->error_count())
                     {
                        numeric_checker_ptr->clear_errors();
                     }
                  }
                  else if (0 != (sequence_validator_ptr = dynamic_cast<lexer::helper::sequence_validator*>(helper_assembly_.error_token_scanner)))
                  {
                     for (std::size_t i = 0; i < sequence_validator_ptr->error_count(); ++i)
                     {
                        std::pair<lexer::token,lexer::token> error_token = sequence_validator_ptr->error(i);

                        set_error(
                           make_error(parser_error::e_token,
                                      error_token.first,
                                      "ERR007 - Invalid token sequence: '" +
                                      error_token.first.value  + "' and '" +
                                      error_token.second.value + "'",
                                      exprtk_error_location));
                     }

                     if (sequence_validator_ptr->error_count())
                     {
                        sequence_validator_ptr->clear_errors();
                     }
                  }
                  else if (0 != (sequence_validator3_ptr = dynamic_cast<lexer::helper::sequence_validator_3tokens*>(helper_assembly_.error_token_scanner)))
                  {
                     for (std::size_t i = 0; i < sequence_validator3_ptr->error_count(); ++i)
                     {
                        std::pair<lexer::token,lexer::token> error_token = sequence_validator3_ptr->error(i);

                        set_error(
                           make_error(parser_error::e_token,
                                      error_token.first,
                                      "ERR008 - Invalid token sequence: '" +
                                      error_token.first.value  + "' and '" +
                                      error_token.second.value + "'",
                                      exprtk_error_location));
                     }

                     if (sequence_validator3_ptr->error_count())
                     {
                        sequence_validator3_ptr->clear_errors();
                     }
                  }
               }

               return false;
            }
         }

         return true;
      }

      inline settings_store& settings()
      {
         return settings_;
      }

      inline parser_error::type get_error(const std::size_t& index) const
      {
         if (index < error_list_.size())
            return error_list_[index];
         else
            throw std::invalid_argument("parser::get_error() - Invalid error index specificed");
      }

      inline std::string error() const
      {
         if (!error_list_.empty())
         {
            return error_list_[0].diagnostic;
         }
         else
            return std::string("No Error");
      }

      inline std::size_t error_count() const
      {
         return error_list_.size();
      }

      inline dependent_entity_collector& dec()
      {
         return dec_;
      }

      inline bool replace_symbol(const std::string& old_symbol, const std::string& new_symbol)
      {
         if (!settings_.replacer_enabled())
            return false;
         else if (details::is_reserved_word(old_symbol))
            return false;
         else
            return symbol_replacer_.add_replace(old_symbol,new_symbol,lexer::token::e_symbol);
      }

      inline bool remove_replace_symbol(const std::string& symbol)
      {
         if (!settings_.replacer_enabled())
            return false;
         else if (details::is_reserved_word(symbol))
            return false;
         else
            return symbol_replacer_.remove(symbol);
      }

      inline void enable_unknown_symbol_resolver(unknown_symbol_resolver* usr = reinterpret_cast<unknown_symbol_resolver*>(0))
      {
         resolve_unknown_symbol_ = true;

         if (usr)
            unknown_symbol_resolver_ = usr;
         else
            unknown_symbol_resolver_ = &default_usr_;
      }

      inline void enable_unknown_symbol_resolver(unknown_symbol_resolver& usr)
      {
         enable_unknown_symbol_resolver(&usr);
      }

      inline void disable_unknown_symbol_resolver()
      {
         resolve_unknown_symbol_  = false;
         unknown_symbol_resolver_ = &default_usr_;
      }

      inline void register_loop_runtime_check(loop_runtime_check& lrtchk)
      {
         loop_runtime_check_ = &lrtchk;
      }

      inline void clear_loop_runtime_check()
      {
         loop_runtime_check_ = loop_runtime_check_ptr(0);
      }

   private:

      inline bool valid_base_operation(const std::string& symbol) const
      {
         const std::size_t length = symbol.size();

         if (
              (length < 3) || // Shortest base op symbol length
              (length > 9)    // Longest base op symbol length
            )
            return false;
         else
            return settings_.function_enabled(symbol) &&
                   (base_ops_map_.end() != base_ops_map_.find(symbol));
      }

      inline bool valid_vararg_operation(const std::string& symbol) const
      {
         static const std::string s_sum     = "sum" ;
         static const std::string s_mul     = "mul" ;
         static const std::string s_avg     = "avg" ;
         static const std::string s_min     = "min" ;
         static const std::string s_max     = "max" ;
         static const std::string s_mand    = "mand";
         static const std::string s_mor     = "mor" ;
         static const std::string s_multi   = "~"   ;
         static const std::string s_mswitch = "[*]" ;

         return
               (
                  details::imatch(symbol,s_sum    ) ||
                  details::imatch(symbol,s_mul    ) ||
                  details::imatch(symbol,s_avg    ) ||
                  details::imatch(symbol,s_min    ) ||
                  details::imatch(symbol,s_max    ) ||
                  details::imatch(symbol,s_mand   ) ||
                  details::imatch(symbol,s_mor    ) ||
                  details::imatch(symbol,s_multi  ) ||
                  details::imatch(symbol,s_mswitch)
               ) &&
               settings_.function_enabled(symbol);
      }

      bool is_invalid_logic_operation(const details::operator_type operation) const
      {
         return settings_.logic_disabled(operation);
      }

      bool is_invalid_arithmetic_operation(const details::operator_type operation) const
      {
         return settings_.arithmetic_disabled(operation);
      }

      bool is_invalid_assignment_operation(const details::operator_type operation) const
      {
         return settings_.assignment_disabled(operation);
      }

      bool is_invalid_inequality_operation(const details::operator_type operation) const
      {
         return settings_.inequality_disabled(operation);
      }

      #ifdef exprtk_enable_debugging
      inline void next_token()
      {
         const std::string ct_str = current_token().value;
         const std::size_t ct_pos = current_token().position;
         parser_helper::next_token();
         const std::string depth(2 * state_.scope_depth,' ');
         exprtk_debug(("%s"
                       "prev[%s | %04d] --> curr[%s | %04d]  stack_level: %3d\n",
                       depth.c_str(),
                       ct_str.c_str(),
                       static_cast<unsigned int>(ct_pos),
                       current_token().value.c_str(),
                       static_cast<unsigned int>(current_token().position),
                       static_cast<unsigned int>(state_.stack_depth)));
      }
      #endif

      inline expression_node_ptr parse_corpus()
      {
         std::vector<expression_node_ptr> arg_list;
         std::vector<bool> side_effect_list;

         scoped_vec_delete<expression_node_t> sdd((*this),arg_list);

         lexer::token begin_token;
         lexer::token end_token;

         for ( ; ; )
         {
            state_.side_effect_present = false;

            begin_token = current_token();

            expression_node_ptr arg = parse_expression();

            if (0 == arg)
            {
               if (error_list_.empty())
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                current_token(),
                                "ERR009 - Invalid expression encountered",
                                exprtk_error_location));
               }

               return error_node();
            }
            else
            {
               arg_list.push_back(arg);

               side_effect_list.push_back(state_.side_effect_present);

               end_token = current_token();

               const std::string sub_expr = construct_subexpr(begin_token, end_token);

               exprtk_debug(("parse_corpus(%02d) Subexpr: %s\n",
                             static_cast<int>(arg_list.size() - 1),
                             sub_expr.c_str()));

               exprtk_debug(("parse_corpus(%02d) - Side effect present: %s\n",
                             static_cast<int>(arg_list.size() - 1),
                             state_.side_effect_present ? "true" : "false"));

               exprtk_debug(("-------------------------------------------------\n"));
            }

            if (lexer().finished())
               break;
            else if (token_is(token_t::e_eof,prsrhlpr_t::e_hold))
            {
               if (lexer().finished())
                  break;
               else
                  next_token();
            }
         }

         if (
              !arg_list.empty() &&
              is_return_node(arg_list.back())
            )
         {
            dec_.final_stmt_return_ = true;
         }

         const expression_node_ptr result = simplify(arg_list,side_effect_list);

         sdd.delete_ptr = (0 == result);

         return result;
      }

      std::string construct_subexpr(lexer::token& begin_token, lexer::token& end_token)
      {
         std::string result = lexer().substr(begin_token.position,end_token.position);

         for (std::size_t i = 0; i < result.size(); ++i)
         {
            if (details::is_whitespace(result[i])) result[i] = ' ';
         }

         return result;
      }

      static const precedence_level default_precedence = e_level00;

      struct state_t
      {
         inline void set(const precedence_level& l,
                         const precedence_level& r,
                         const details::operator_type& o)
         {
            left  = l;
            right = r;
            operation = o;
         }

         inline void reset()
         {
            left      = e_level00;
            right     = e_level00;
            operation = details::e_default;
         }

         precedence_level left;
         precedence_level right;
         details::operator_type operation;
      };

      inline expression_node_ptr parse_expression(precedence_level precedence = e_level00)
      {
         stack_limit_handler slh(*this);

         if (!slh)
         {
            return error_node();
         }

         expression_node_ptr expression = parse_branch(precedence);

         if (0 == expression)
         {
            return error_node();
         }

         bool break_loop = false;

         state_t current_state;

         for ( ; ; )
         {
            current_state.reset();

            switch (current_token().type)
            {
               case token_t::e_assign : current_state.set(e_level00, e_level00, details::e_assign); break;
               case token_t::e_addass : current_state.set(e_level00, e_level00, details::e_addass); break;
               case token_t::e_subass : current_state.set(e_level00, e_level00, details::e_subass); break;
               case token_t::e_mulass : current_state.set(e_level00, e_level00, details::e_mulass); break;
               case token_t::e_divass : current_state.set(e_level00, e_level00, details::e_divass); break;
               case token_t::e_modass : current_state.set(e_level00, e_level00, details::e_modass); break;
               case token_t::e_swap   : current_state.set(e_level00, e_level00, details::e_swap  ); break;
               case token_t::e_lt     : current_state.set(e_level05, e_level06, details::e_lt    ); break;
               case token_t::e_lte    : current_state.set(e_level05, e_level06, details::e_lte   ); break;
               case token_t::e_eq     : current_state.set(e_level05, e_level06, details::e_eq    ); break;
               case token_t::e_ne     : current_state.set(e_level05, e_level06, details::e_ne    ); break;
               case token_t::e_gte    : current_state.set(e_level05, e_level06, details::e_gte   ); break;
               case token_t::e_gt     : current_state.set(e_level05, e_level06, details::e_gt    ); break;
               case token_t::e_add    : current_state.set(e_level07, e_level08, details::e_add   ); break;
               case token_t::e_sub    : current_state.set(e_level07, e_level08, details::e_sub   ); break;
               case token_t::e_div    : current_state.set(e_level10, e_level11, details::e_div   ); break;
               case token_t::e_mul    : current_state.set(e_level10, e_level11, details::e_mul   ); break;
               case token_t::e_mod    : current_state.set(e_level10, e_level11, details::e_mod   ); break;
               case token_t::e_pow    : current_state.set(e_level12, e_level12, details::e_pow   ); break;
               default                : if (token_t::e_symbol == current_token().type)
                                        {
                                           static const std::string s_and   = "and"  ;
                                           static const std::string s_nand  = "nand" ;
                                           static const std::string s_or    = "or"   ;
                                           static const std::string s_nor   = "nor"  ;
                                           static const std::string s_xor   = "xor"  ;
                                           static const std::string s_xnor  = "xnor" ;
                                           static const std::string s_in    = "in"   ;
                                           static const std::string s_like  = "like" ;
                                           static const std::string s_ilike = "ilike";
                                           static const std::string s_and1  = "&"    ;
                                           static const std::string s_or1   = "|"    ;
                                           static const std::string s_not   = "not"  ;

                                           if (details::imatch(current_token().value,s_and))
                                           {
                                              current_state.set(e_level03, e_level04, details::e_and);
                                              break;
                                           }
                                           else if (details::imatch(current_token().value,s_and1))
                                           {
                                              current_state.set(e_level03, e_level04, details::e_and);
                                              break;
                                           }
                                           else if (details::imatch(current_token().value,s_nand))
                                           {
                                              current_state.set(e_level03, e_level04, details::e_nand);
                                              break;
                                           }
                                           else if (details::imatch(current_token().value,s_or))
                                           {
                                              current_state.set(e_level01, e_level02, details::e_or);
                                              break;
                                           }
                                           else if (details::imatch(current_token().value,s_or1))
                                           {
                                              current_state.set(e_level01, e_level02, details::e_or);
                                              break;
                                           }
                                           else if (details::imatch(current_token().value,s_nor))
                                           {
                                              current_state.set(e_level01, e_level02, details::e_nor);
                                              break;
                                           }
                                           else if (details::imatch(current_token().value,s_xor))
                                           {
                                              current_state.set(e_level01, e_level02, details::e_xor);
                                              break;
                                           }
                                           else if (details::imatch(current_token().value,s_xnor))
                                           {
                                              current_state.set(e_level01, e_level02, details::e_xnor);
                                              break;
                                           }
                                           else if (details::imatch(current_token().value,s_in))
                                           {
                                              current_state.set(e_level04, e_level04, details::e_in);
                                              break;
                                           }
                                           else if (details::imatch(current_token().value,s_like))
                                           {
                                              current_state.set(e_level04, e_level04, details::e_like);
                                              break;
                                           }
                                           else if (details::imatch(current_token().value,s_ilike))
                                           {
                                              current_state.set(e_level04, e_level04, details::e_ilike);
                                              break;
                                           }
                                           else if (details::imatch(current_token().value,s_not))
                                           {
                                              break;
                                           }
                                        }

                                        break_loop = true;
            }

            if (break_loop)
            {
               parse_pending_string_rangesize(expression);
               break;
            }
            else if (current_state.left < precedence)
               break;

            const lexer::token prev_token = current_token();

            next_token();

            expression_node_ptr right_branch   = error_node();
            expression_node_ptr new_expression = error_node();

            if (is_invalid_logic_operation(current_state.operation))
            {
               free_node(node_allocator_,expression);

               set_error(
                  make_error(parser_error::e_syntax,
                             prev_token,
                             "ERR010 - Invalid or disabled logic operation '" + details::to_str(current_state.operation) + "'",
                             exprtk_error_location));

               return error_node();
            }
            else if (is_invalid_arithmetic_operation(current_state.operation))
            {
               free_node(node_allocator_,expression);

               set_error(
                  make_error(parser_error::e_syntax,
                             prev_token,
                             "ERR011 - Invalid or disabled arithmetic operation '" + details::to_str(current_state.operation) + "'",
                             exprtk_error_location));

               return error_node();
            }
            else if (is_invalid_inequality_operation(current_state.operation))
            {
               free_node(node_allocator_,expression);

               set_error(
                  make_error(parser_error::e_syntax,
                             prev_token,
                             "ERR012 - Invalid inequality operation '" + details::to_str(current_state.operation) + "'",
                             exprtk_error_location));

               return error_node();
            }
            else if (is_invalid_assignment_operation(current_state.operation))
            {
               free_node(node_allocator_,expression);

               set_error(
                  make_error(parser_error::e_syntax,
                             prev_token,
                             "ERR013 - Invalid or disabled assignment operation '" + details::to_str(current_state.operation) + "'",
                             exprtk_error_location));

               return error_node();
            }

            if (0 != (right_branch = parse_expression(current_state.right)))
            {
               if (
                    details::is_return_node(expression  ) ||
                    details::is_return_node(right_branch)
                  )
               {
                  free_node(node_allocator_, expression  );
                  free_node(node_allocator_, right_branch);

                  set_error(
                     make_error(parser_error::e_syntax,
                                prev_token,
                                "ERR014 - Return statements cannot be part of sub-expressions",
                                exprtk_error_location));

                  return error_node();
               }

               new_expression = expression_generator_
                                  (
                                    current_state.operation,
                                    expression,
                                    right_branch
                                  );
            }

            if (0 == new_expression)
            {
               if (error_list_.empty())
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                prev_token,
                                !synthesis_error_.empty() ?
                                synthesis_error_ :
                                "ERR015 - General parsing error at token: '" + prev_token.value + "'",
                                exprtk_error_location));
               }

               free_node(node_allocator_, expression  );
               free_node(node_allocator_, right_branch);

               return error_node();
            }
            else
            {
               if (
                    token_is(token_t::e_ternary,prsrhlpr_t::e_hold) &&
                    (e_level00 == precedence)
                  )
               {
                  expression = parse_ternary_conditional_statement(new_expression);
               }
               else
                  expression = new_expression;

               parse_pending_string_rangesize(expression);
            }
         }

         if ((0 != expression) && (expression->node_depth() > settings_.max_node_depth_))
         {
            set_error(
               make_error(parser_error::e_syntax,
                  current_token(),
                  "ERR016 - Expression depth of " + details::to_str(static_cast<int>(expression->node_depth())) +
                  " exceeds maximum allowed expression depth of " + details::to_str(static_cast<int>(settings_.max_node_depth_)),
                  exprtk_error_location));

            free_node(node_allocator_,expression);

            return error_node();
         }

         return expression;
      }

      bool simplify_unary_negation_branch(expression_node_ptr& node)
      {
         {
            typedef details::unary_branch_node<T,details::neg_op<T> > ubn_t;
            ubn_t* n = dynamic_cast<ubn_t*>(node);

            if (n)
            {
               expression_node_ptr un_r = n->branch(0);
               n->release();
               free_node(node_allocator_,node);
               node = un_r;

               return true;
            }
         }

         {
            typedef details::unary_variable_node<T,details::neg_op<T> > uvn_t;

            uvn_t* n = dynamic_cast<uvn_t*>(node);

            if (n)
            {
               const T& v = n->v();
               expression_node_ptr return_node = error_node();

               if (
                    (0 != (return_node = symtab_store_.get_variable(v))) ||
                    (0 != (return_node = sem_         .get_variable(v)))
                  )
               {
                  free_node(node_allocator_,node);
                  node = return_node;

                  return true;
               }
               else
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                current_token(),
                                "ERR017 - Failed to find variable node in symbol table",
                                exprtk_error_location));

                  free_node(node_allocator_,node);

                  return false;
               }
            }
         }

         return false;
      }

      static inline expression_node_ptr error_node()
      {
         return reinterpret_cast<expression_node_ptr>(0);
      }

      struct scoped_expression_delete
      {
         scoped_expression_delete(parser<T>& pr, expression_node_ptr& expression)
         : delete_ptr(true)
         , parser_(pr)
         , expression_(expression)
         {}

        ~scoped_expression_delete()
         {
            if (delete_ptr)
            {
               free_node(parser_.node_allocator_, expression_);
            }
         }

         bool delete_ptr;
         parser<T>& parser_;
         expression_node_ptr& expression_;

      private:

         scoped_expression_delete(const scoped_expression_delete&) exprtk_delete;
         scoped_expression_delete& operator=(const scoped_expression_delete&) exprtk_delete;
      };

      template <typename Type, std::size_t N>
      struct scoped_delete
      {
         typedef Type* ptr_t;

         scoped_delete(parser<T>& pr, ptr_t& p)
         : delete_ptr(true)
         , parser_(pr)
         , p_(&p)
         {}

         scoped_delete(parser<T>& pr, ptr_t (&p)[N])
         : delete_ptr(true)
         , parser_(pr)
         , p_(&p[0])
         {}

        ~scoped_delete()
         {
            if (delete_ptr)
            {
               for (std::size_t i = 0; i < N; ++i)
               {
                  free_node(parser_.node_allocator_, p_[i]);
               }
            }
         }

         bool delete_ptr;
         parser<T>& parser_;
         ptr_t* p_;

      private:

         scoped_delete(const scoped_delete<Type,N>&) exprtk_delete;
         scoped_delete<Type,N>& operator=(const scoped_delete<Type,N>&) exprtk_delete;
      };

      template <typename Type>
      struct scoped_deq_delete
      {
         typedef Type* ptr_t;

         scoped_deq_delete(parser<T>& pr, std::deque<ptr_t>& deq)
         : delete_ptr(true)
         , parser_(pr)
         , deq_(deq)
         {}

        ~scoped_deq_delete()
         {
            if (delete_ptr && !deq_.empty())
            {
               for (std::size_t i = 0; i < deq_.size(); ++i)
               {
                  free_node(parser_.node_allocator_,deq_[i]);
               }

               deq_.clear();
            }
         }

         bool delete_ptr;
         parser<T>& parser_;
         std::deque<ptr_t>& deq_;

      private:

         scoped_deq_delete(const scoped_deq_delete<Type>&) exprtk_delete;
         scoped_deq_delete<Type>& operator=(const scoped_deq_delete<Type>&) exprtk_delete;
      };

      template <typename Type>
      struct scoped_vec_delete
      {
         typedef Type* ptr_t;

         scoped_vec_delete(parser<T>& pr, std::vector<ptr_t>& vec)
         : delete_ptr(true)
         , parser_(pr)
         , vec_(vec)
         {}

        ~scoped_vec_delete()
         {
            if (delete_ptr && !vec_.empty())
            {
               for (std::size_t i = 0; i < vec_.size(); ++i)
               {
                  free_node(parser_.node_allocator_,vec_[i]);
               }

               vec_.clear();
            }
         }

         bool delete_ptr;
         parser<T>& parser_;
         std::vector<ptr_t>& vec_;

      private:

         scoped_vec_delete(const scoped_vec_delete<Type>&) exprtk_delete;
         scoped_vec_delete<Type>& operator=(const scoped_vec_delete<Type>&) exprtk_delete;
      };

      struct scoped_bool_negator
      {
         explicit scoped_bool_negator(bool& bb)
         : b(bb)
         { b = !b; }

        ~scoped_bool_negator()
         { b = !b; }

         bool& b;
      };

      struct scoped_bool_or_restorer
      {
         explicit scoped_bool_or_restorer(bool& bb)
         : b(bb)
         , original_value_(bb)
         {}

        ~scoped_bool_or_restorer()
         {
            b = b || original_value_;
         }

         bool& b;
         bool original_value_;
      };

      struct scoped_inc_dec
      {
         explicit scoped_inc_dec(std::size_t& v)
         : v_(v)
         { ++v_; }

        ~scoped_inc_dec()
         {
           assert(v_ > 0);
           --v_;
         }

         std::size_t& v_;
      };

      inline expression_node_ptr parse_function_invocation(ifunction<T>* function, const std::string& function_name)
      {
         expression_node_ptr func_node = reinterpret_cast<expression_node_ptr>(0);

         switch (function->param_count)
         {
            case  0 : func_node = parse_function_call_0  (function,function_name); break;
            case  1 : func_node = parse_function_call< 1>(function,function_name); break;
            case  2 : func_node = parse_function_call< 2>(function,function_name); break;
            case  3 : func_node = parse_function_call< 3>(function,function_name); break;
            case  4 : func_node = parse_function_call< 4>(function,function_name); break;
            case  5 : func_node = parse_function_call< 5>(function,function_name); break;
            case  6 : func_node = parse_function_call< 6>(function,function_name); break;
            case  7 : func_node = parse_function_call< 7>(function,function_name); break;
            case  8 : func_node = parse_function_call< 8>(function,function_name); break;
            case  9 : func_node = parse_function_call< 9>(function,function_name); break;
            case 10 : func_node = parse_function_call<10>(function,function_name); break;
            case 11 : func_node = parse_function_call<11>(function,function_name); break;
            case 12 : func_node = parse_function_call<12>(function,function_name); break;
            case 13 : func_node = parse_function_call<13>(function,function_name); break;
            case 14 : func_node = parse_function_call<14>(function,function_name); break;
            case 15 : func_node = parse_function_call<15>(function,function_name); break;
            case 16 : func_node = parse_function_call<16>(function,function_name); break;
            case 17 : func_node = parse_function_call<17>(function,function_name); break;
            case 18 : func_node = parse_function_call<18>(function,function_name); break;
            case 19 : func_node = parse_function_call<19>(function,function_name); break;
            case 20 : func_node = parse_function_call<20>(function,function_name); break;
            default : {
                         set_error(
                            make_error(parser_error::e_syntax,
                                       current_token(),
                                       "ERR018 - Invalid number of parameters for function: '" + function_name + "'",
                                       exprtk_error_location));

                         return error_node();
                      }
         }

         if (func_node)
            return func_node;
         else
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR019 - Failed to generate call to function: '" + function_name + "'",
                          exprtk_error_location));

            return error_node();
         }
      }

      template <std::size_t NumberofParameters>
      inline expression_node_ptr parse_function_call(ifunction<T>* function, const std::string& function_name)
      {
         #ifdef _MSC_VER
            #pragma warning(push)
            #pragma warning(disable: 4127)
         #endif
         if (0 == NumberofParameters)
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR020 - Expecting ifunction '" + function_name + "' to have non-zero parameter count",
                          exprtk_error_location));

            return error_node();
         }
         #ifdef _MSC_VER
            #pragma warning(pop)
         #endif

         expression_node_ptr branch[NumberofParameters];
         expression_node_ptr result  = error_node();

         std::fill_n(branch, NumberofParameters, reinterpret_cast<expression_node_ptr>(0));

         scoped_delete<expression_node_t,NumberofParameters> sd((*this),branch);

         next_token();

         if (!token_is(token_t::e_lbracket))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR021 - Expecting argument list for function: '" + function_name + "'",
                          exprtk_error_location));

            return error_node();
         }

         for (int i = 0; i < static_cast<int>(NumberofParameters); ++i)
         {
            branch[i] = parse_expression();

            if (0 == branch[i])
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR022 - Failed to parse argument " + details::to_str(i) + " for function: '" + function_name + "'",
                             exprtk_error_location));

               return error_node();
            }
            else if (i < static_cast<int>(NumberofParameters - 1))
            {
               if (!token_is(token_t::e_comma))
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                current_token(),
                                "ERR023 - Invalid number of arguments for function: '" + function_name + "'",
                                exprtk_error_location));

                  return error_node();
               }
            }
         }

         if (!token_is(token_t::e_rbracket))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR024 - Invalid number of arguments for function: '" + function_name + "'",
                          exprtk_error_location));

            return error_node();
         }
         else
            result = expression_generator_.function(function,branch);

         sd.delete_ptr = (0 == result);

         return result;
      }

      inline expression_node_ptr parse_function_call_0(ifunction<T>* function, const std::string& function_name)
      {
         expression_node_ptr result = expression_generator_.function(function);

         state_.side_effect_present = function->has_side_effects();

         next_token();

         if (
               token_is(token_t::e_lbracket) &&
              !token_is(token_t::e_rbracket)
            )
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR025 - Expecting '()' to proceed call to function: '" + function_name + "'",
                          exprtk_error_location));

            free_node(node_allocator_,result);

            return error_node();
         }
         else
            return result;
      }

      template <std::size_t MaxNumberofParameters>
      inline std::size_t parse_base_function_call(expression_node_ptr (&param_list)[MaxNumberofParameters], const std::string& function_name = "")
      {
         std::fill_n(param_list, MaxNumberofParameters, reinterpret_cast<expression_node_ptr>(0));

         scoped_delete<expression_node_t,MaxNumberofParameters> sd((*this),param_list);

         next_token();

         if (!token_is(token_t::e_lbracket))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR026 - Expected a '(' at start of function call to '" + function_name  +
                          "', instead got: '" + current_token().value + "'",
                          exprtk_error_location));

            return 0;
         }

         if (token_is(token_t::e_rbracket, e_hold))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR027 - Expected at least one input parameter for function call '" + function_name + "'",
                          exprtk_error_location));

            return 0;
         }

         std::size_t param_index = 0;

         for (; param_index < MaxNumberofParameters; ++param_index)
         {
            param_list[param_index] = parse_expression();

            if (0 == param_list[param_index])
               return 0;
            else if (token_is(token_t::e_rbracket))
            {
               sd.delete_ptr = false;
               break;
            }
            else if (token_is(token_t::e_comma))
               continue;
            else
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR028 - Expected a ',' between function input parameters, instead got: '" + current_token().value + "'",
                             exprtk_error_location));

               return 0;
            }
         }

         if (sd.delete_ptr)
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR029 - Invalid number of input parameters passed to function '" + function_name  + "'",
                          exprtk_error_location));

            return 0;
         }

         return (param_index + 1);
      }

      inline expression_node_ptr parse_base_operation()
      {
         typedef std::pair<base_ops_map_t::iterator,base_ops_map_t::iterator> map_range_t;

         const std::string operation_name   = current_token().value;
         const token_t     diagnostic_token = current_token();

         map_range_t itr_range = base_ops_map_.equal_range(operation_name);

         if (0 == std::distance(itr_range.first,itr_range.second))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          diagnostic_token,
                          "ERR030 - No entry found for base operation: " + operation_name,
                          exprtk_error_location));

            return error_node();
         }

         static const std::size_t MaxNumberofParameters = 4;
         expression_node_ptr param_list[MaxNumberofParameters] = {0};

         const std::size_t parameter_count = parse_base_function_call(param_list, operation_name);

         if ((parameter_count > 0) && (parameter_count <= MaxNumberofParameters))
         {
            for (base_ops_map_t::iterator itr = itr_range.first; itr != itr_range.second; ++itr)
            {
               const details::base_operation_t& operation = itr->second;

               if (operation.num_params == parameter_count)
               {
                  switch (parameter_count)
                  {
                     #define base_opr_case(N)                                         \
                     case N : {                                                       \
                                 expression_node_ptr pl##N[N] = {0};                  \
                                 std::copy(param_list, param_list + N, pl##N);        \
                                 lodge_symbol(operation_name, e_st_function);         \
                                 return expression_generator_(operation.type, pl##N); \
                              }                                                       \

                     base_opr_case(1)
                     base_opr_case(2)
                     base_opr_case(3)
                     base_opr_case(4)
                     #undef base_opr_case
                  }
               }
            }
         }

         for (std::size_t i = 0; i < MaxNumberofParameters; ++i)
         {
            free_node(node_allocator_, param_list[i]);
         }

         set_error(
            make_error(parser_error::e_syntax,
                       diagnostic_token,
                       "ERR031 - Invalid number of input parameters for call to function: '" + operation_name + "'",
                       exprtk_error_location));

         return error_node();
      }

      inline expression_node_ptr parse_conditional_statement_01(expression_node_ptr condition)
      {
         // Parse: [if][(][condition][,][consequent][,][alternative][)]

         expression_node_ptr consequent  = error_node();
         expression_node_ptr alternative = error_node();

         bool result = true;

         if (!token_is(token_t::e_comma))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR032 - Expected ',' between if-statement condition and consequent",
                          exprtk_error_location));
            result = false;
         }
         else if (0 == (consequent = parse_expression()))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR033 - Failed to parse consequent for if-statement",
                          exprtk_error_location));
            result = false;
         }
         else if (!token_is(token_t::e_comma))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR034 - Expected ',' between if-statement consequent and alternative",
                          exprtk_error_location));
            result = false;
         }
         else if (0 == (alternative = parse_expression()))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR035 - Failed to parse alternative for if-statement",
                          exprtk_error_location));
            result = false;
         }
         else if (!token_is(token_t::e_rbracket))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR036 - Expected ')' at the end of if-statement",
                          exprtk_error_location));
            result = false;
         }

         if (result)
         {
            const bool consq_is_vec = is_ivector_node(consequent );
            const bool alter_is_vec = is_ivector_node(alternative);

            if (consq_is_vec || alter_is_vec)
            {
               if (consq_is_vec && alter_is_vec)
               {
                  return expression_generator_
                           .conditional_vector(condition, consequent, alternative);
               }

               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR038 - Return types of if-statement differ: vector/non-vector",
                             exprtk_error_location));

               result = false;
            }
         }

         if (!result)
         {
            free_node(node_allocator_, condition  );
            free_node(node_allocator_, consequent );
            free_node(node_allocator_, alternative);

            return error_node();
         }
         else
            return expression_generator_
                      .conditional(condition, consequent, alternative);
      }

      inline expression_node_ptr parse_conditional_statement_02(expression_node_ptr condition)
      {
         expression_node_ptr consequent  = error_node();
         expression_node_ptr alternative = error_node();

         bool result = true;

         if (token_is(token_t::e_lcrlbracket,prsrhlpr_t::e_hold))
         {
            if (0 == (consequent = parse_multi_sequence("if-statement-01")))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR039 - Failed to parse body of consequent for if-statement",
                             exprtk_error_location));

               result = false;
            }
         }
         else
         {
            if (
                 settings_.commutative_check_enabled() &&
                 token_is(token_t::e_mul,prsrhlpr_t::e_hold)
               )
            {
               next_token();
            }

            if (0 != (consequent = parse_expression()))
            {
               if (!token_is(token_t::e_eof))
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                current_token(),
                                "ERR040 - Expected ';' at the end of the consequent for if-statement",
                                exprtk_error_location));

                  result = false;
               }
            }
            else
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR041 - Failed to parse body of consequent for if-statement",
                             exprtk_error_location));

               result = false;
            }
         }

         if (result)
         {
            if (details::imatch(current_token().value,"else"))
            {
               next_token();

               if (token_is(token_t::e_lcrlbracket,prsrhlpr_t::e_hold))
               {
                  if (0 == (alternative = parse_multi_sequence("else-statement-01")))
                  {
                     set_error(
                        make_error(parser_error::e_syntax,
                                   current_token(),
                                   "ERR042 - Failed to parse body of the 'else' for if-statement",
                                   exprtk_error_location));

                     result = false;
                  }
               }
               else if (details::imatch(current_token().value,"if"))
               {
                  if (0 == (alternative = parse_conditional_statement()))
                  {
                     set_error(
                        make_error(parser_error::e_syntax,
                                   current_token(),
                                   "ERR043 - Failed to parse body of if-else statement",
                                   exprtk_error_location));

                     result = false;
                  }
               }
               else if (0 != (alternative = parse_expression()))
               {
                  if (!token_is(token_t::e_eof))
                  {
                     set_error(
                        make_error(parser_error::e_syntax,
                                   current_token(),
                                   "ERR044 - Expected ';' at the end of the 'else-if' for the if-statement",
                                   exprtk_error_location));

                     result = false;
                  }
               }
               else
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                current_token(),
                                "ERR045 - Failed to parse body of the 'else' for if-statement",
                                exprtk_error_location));

                  result = false;
               }
            }
         }

         if (result)
         {
            const bool consq_is_vec = is_ivector_node(consequent );
            const bool alter_is_vec = is_ivector_node(alternative);

            if (consq_is_vec || alter_is_vec)
            {
               if (consq_is_vec && alter_is_vec)
               {
                  return expression_generator_
                           .conditional_vector(condition, consequent, alternative);
               }

               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR047 - Return types of if-statement differ: vector/non-vector",
                             exprtk_error_location));

               result = false;
            }
         }

         if (!result)
         {
            free_node(node_allocator_, condition  );
            free_node(node_allocator_, consequent );
            free_node(node_allocator_, alternative);

            return error_node();
         }
         else
            return expression_generator_
                      .conditional(condition, consequent, alternative);
      }

      inline expression_node_ptr parse_conditional_statement()
      {
         expression_node_ptr condition = error_node();

         next_token();

         if (!token_is(token_t::e_lbracket))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR048 - Expected '(' at start of if-statement, instead got: '" + current_token().value + "'",
                          exprtk_error_location));

            return error_node();
         }
         else if (0 == (condition = parse_expression()))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR049 - Failed to parse condition for if-statement",
                          exprtk_error_location));

            return error_node();
         }
         else if (token_is(token_t::e_comma,prsrhlpr_t::e_hold))
         {
            // if (x,y,z)
            return parse_conditional_statement_01(condition);
         }
         else if (token_is(token_t::e_rbracket))
         {
            /*
               00. if (x) y;
               01. if (x) y; else z;
               02. if (x) y; else {z0; ... zn;}
               03. if (x) y; else if (z) w;
               04. if (x) y; else if (z) w; else u;
               05. if (x) y; else if (z) w; else {u0; ... un;}
               06. if (x) y; else if (z) {w0; ... wn;}
               07. if (x) {y0; ... yn;}
               08. if (x) {y0; ... yn;} else z;
               09. if (x) {y0; ... yn;} else {z0; ... zn;};
               10. if (x) {y0; ... yn;} else if (z) w;
               11. if (x) {y0; ... yn;} else if (z) w; else u;
               12. if (x) {y0; ... nex;} else if (z) w; else {u0 ... un;}
               13. if (x) {y0; ... yn;} else if (z) {w0; ... wn;}
            */
            return parse_conditional_statement_02(condition);
         }

         set_error(
            make_error(parser_error::e_syntax,
                       current_token(),
                       "ERR050 - Invalid if-statement",
                       exprtk_error_location));

         free_node(node_allocator_,condition);

         return error_node();
      }

      inline expression_node_ptr parse_ternary_conditional_statement(expression_node_ptr condition)
      {
         // Parse: [condition][?][consequent][:][alternative]
         expression_node_ptr consequent  = error_node();
         expression_node_ptr alternative = error_node();

         bool result = true;

         if (0 == condition)
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR051 - Encountered invalid condition branch for ternary if-statement",
                          exprtk_error_location));

            return error_node();
         }
         else if (!token_is(token_t::e_ternary))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR052 - Expected '?' after condition of ternary if-statement",
                          exprtk_error_location));

            result = false;
         }
         else if (0 == (consequent = parse_expression()))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR053 - Failed to parse consequent for ternary if-statement",
                          exprtk_error_location));

            result = false;
         }
         else if (!token_is(token_t::e_colon))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR054 - Expected ':' between ternary if-statement consequent and alternative",
                          exprtk_error_location));

            result = false;
         }
         else if (0 == (alternative = parse_expression()))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR055 - Failed to parse alternative for ternary if-statement",
                          exprtk_error_location));

            result = false;
         }

         if (result)
         {
            const bool consq_is_vec = is_ivector_node(consequent );
            const bool alter_is_vec = is_ivector_node(alternative);

            if (consq_is_vec || alter_is_vec)
            {
               if (consq_is_vec && alter_is_vec)
               {
                  return expression_generator_
                           .conditional_vector(condition, consequent, alternative);
               }

               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR057 - Return types of ternary differ: vector/non-vector",
                             exprtk_error_location));

               result = false;
            }
         }

         if (!result)
         {
            free_node(node_allocator_, condition  );
            free_node(node_allocator_, consequent );
            free_node(node_allocator_, alternative);

            return error_node();
         }
         else
            return expression_generator_
                      .conditional(condition, consequent, alternative);
      }

      inline expression_node_ptr parse_not_statement()
      {
         if (settings_.logic_disabled("not"))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR058 - Invalid or disabled logic operation 'not'",
                          exprtk_error_location));

            return error_node();
         }

         return parse_base_operation();
      }

      inline expression_node_ptr parse_while_loop()
      {
         // Parse: [while][(][test expr][)][{][expression][}]
         expression_node_ptr condition   = error_node();
         expression_node_ptr branch      = error_node();
         expression_node_ptr result_node = error_node();

         bool result = true;

         next_token();

         if (!token_is(token_t::e_lbracket))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR059 - Expected '(' at start of while-loop condition statement",
                          exprtk_error_location));

            return error_node();
         }
         else if (0 == (condition = parse_expression()))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR060 - Failed to parse condition for while-loop",
                          exprtk_error_location));

            return error_node();
         }
         else if (!token_is(token_t::e_rbracket))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR061 - Expected ')' at end of while-loop condition statement",
                          exprtk_error_location));

            result = false;
         }

         brkcnt_list_.push_front(false);

         if (result)
         {
            scoped_inc_dec sid(state_.parsing_loop_stmt_count);

            if (0 == (branch = parse_multi_sequence("while-loop")))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR062 - Failed to parse body of while-loop"));
               result = false;
            }
            else if (0 == (result_node = expression_generator_.while_loop(condition,
                                                                          branch,
                                                                          brkcnt_list_.front())))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR063 - Failed to synthesize while-loop",
                             exprtk_error_location));

               result = false;
            }
         }

         if (!result)
         {
            free_node(node_allocator_, branch     );
            free_node(node_allocator_, condition  );
            free_node(node_allocator_, result_node);

            brkcnt_list_.pop_front();

            return error_node();
         }
         else
            return result_node;
      }

      inline expression_node_ptr parse_repeat_until_loop()
      {
         // Parse: [repeat][{][expression][}][until][(][test expr][)]
         expression_node_ptr condition = error_node();
         expression_node_ptr branch    = error_node();
         next_token();

         std::vector<expression_node_ptr> arg_list;
         std::vector<bool> side_effect_list;

         scoped_vec_delete<expression_node_t> sdd((*this),arg_list);

         brkcnt_list_.push_front(false);

         if (details::imatch(current_token().value,"until"))
         {
            next_token();
            branch = node_allocator_.allocate<details::null_node<T> >();
         }
         else
         {
            const token_t::token_type seperator = token_t::e_eof;

            scope_handler sh(*this);

            scoped_bool_or_restorer sbr(state_.side_effect_present);

            scoped_inc_dec sid(state_.parsing_loop_stmt_count);

            for ( ; ; )
            {
               state_.side_effect_present = false;

               expression_node_ptr arg = parse_expression();

               if (0 == arg)
                  return error_node();
               else
               {
                  arg_list.push_back(arg);
                  side_effect_list.push_back(state_.side_effect_present);
               }

               if (details::imatch(current_token().value,"until"))
               {
                  next_token();
                  break;
               }

               const bool is_next_until = peek_token_is(token_t::e_symbol) &&
                                          peek_token_is("until");

               if (!token_is(seperator) && is_next_until)
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                current_token(),
                                "ERR064 - Expected '" + token_t::to_str(seperator) + "' in body of repeat until loop",
                                exprtk_error_location));

                  return error_node();
               }

               if (details::imatch(current_token().value,"until"))
               {
                  next_token();
                  break;
               }
            }

            branch = simplify(arg_list,side_effect_list);

            sdd.delete_ptr = (0 == branch);

            if (sdd.delete_ptr)
            {
               brkcnt_list_.pop_front();

               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR065 - Failed to parse body of repeat until loop",
                             exprtk_error_location));

               return error_node();
            }
         }

         if (!token_is(token_t::e_lbracket))
         {
            brkcnt_list_.pop_front();

            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR066 - Expected '(' before condition statement of repeat until loop",
                          exprtk_error_location));

            free_node(node_allocator_,branch);

            return error_node();
         }
         else if (0 == (condition = parse_expression()))
         {
            brkcnt_list_.pop_front();

            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR067 - Failed to parse condition for repeat until loop",
                          exprtk_error_location));

            free_node(node_allocator_,branch);

            return error_node();
         }
         else if (!token_is(token_t::e_rbracket))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR068 - Expected ')' after condition of repeat until loop",
                          exprtk_error_location));

            free_node(node_allocator_, branch   );
            free_node(node_allocator_, condition);

            brkcnt_list_.pop_front();

            return error_node();
         }

         expression_node_ptr result;

         result = expression_generator_
                     .repeat_until_loop(condition, branch, brkcnt_list_.front());

         if (0 == result)
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR069 - Failed to synthesize repeat until loop",
                          exprtk_error_location));

            free_node(node_allocator_,condition);

            brkcnt_list_.pop_front();

            return error_node();
         }
         else
         {
            brkcnt_list_.pop_front();
            return result;
         }
      }

      inline expression_node_ptr parse_for_loop()
      {
         expression_node_ptr initialiser = error_node();
         expression_node_ptr condition   = error_node();
         expression_node_ptr incrementor = error_node();
         expression_node_ptr loop_body   = error_node();

         scope_element* se = 0;
         bool result       = true;

         next_token();

         scope_handler sh(*this);

         if (!token_is(token_t::e_lbracket))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR070 - Expected '(' at start of for-loop",
                          exprtk_error_location));

            return error_node();
         }

         if (!token_is(token_t::e_eof))
         {
            if (
                 !token_is(token_t::e_symbol,prsrhlpr_t::e_hold) &&
                 details::imatch(current_token().value,"var")
               )
            {
               next_token();

               if (!token_is(token_t::e_symbol,prsrhlpr_t::e_hold))
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                current_token(),
                                "ERR071 - Expected a variable at the start of initialiser section of for-loop",
                                exprtk_error_location));

                  return error_node();
               }
               else if (!peek_token_is(token_t::e_assign))
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                current_token(),
                                "ERR072 - Expected variable assignment of initialiser section of for-loop",
                                exprtk_error_location));

                  return error_node();
               }

               const std::string loop_counter_symbol = current_token().value;

               se = &sem_.get_element(loop_counter_symbol);

               if ((se->name == loop_counter_symbol) && se->active)
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                current_token(),
                                "ERR073 - For-loop variable '" + loop_counter_symbol+ "' is being shadowed by a previous declaration",
                                exprtk_error_location));

                  return error_node();
               }
               else if (!symtab_store_.is_variable(loop_counter_symbol))
               {
                  if (
                       !se->active &&
                       (se->name == loop_counter_symbol) &&
                       (se->type == scope_element::e_variable)
                     )
                  {
                     se->active = true;
                     se->ref_count++;
                  }
                  else
                  {
                     scope_element nse;
                     nse.name      = loop_counter_symbol;
                     nse.active    = true;
                     nse.ref_count = 1;
                     nse.type      = scope_element::e_variable;
                     nse.depth     = state_.scope_depth;
                     nse.data      = new T(T(0));
                     nse.var_node  = node_allocator_.allocate<variable_node_t>(*reinterpret_cast<T*>(nse.data));

                     if (!sem_.add_element(nse))
                     {
                        set_error(
                           make_error(parser_error::e_syntax,
                                      current_token(),
                                      "ERR074 - Failed to add new local variable '" + loop_counter_symbol + "' to SEM",
                                      exprtk_error_location));

                        sem_.free_element(nse);

                        result = false;
                     }
                     else
                     {
                        exprtk_debug(("parse_for_loop() - INFO - Added new local variable: %s\n",nse.name.c_str()));

                        state_.activate_side_effect("parse_for_loop()");
                     }
                  }
               }
            }

            if (0 == (initialiser = parse_expression()))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR075 - Failed to parse initialiser of for-loop",
                             exprtk_error_location));

               result = false;
            }
            else if (!token_is(token_t::e_eof))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR076 - Expected ';' after initialiser of for-loop",
                             exprtk_error_location));

               result = false;
            }
         }

         if (!token_is(token_t::e_eof))
         {
            if (0 == (condition = parse_expression()))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR077 - Failed to parse condition of for-loop",
                             exprtk_error_location));

               result = false;
            }
            else if (!token_is(token_t::e_eof))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR078 - Expected ';' after condition section of for-loop",
                             exprtk_error_location));

               result = false;
            }
         }

         if (!token_is(token_t::e_rbracket))
         {
            if (0 == (incrementor = parse_expression()))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR079 - Failed to parse incrementor of for-loop",
                             exprtk_error_location));

               result = false;
            }
            else if (!token_is(token_t::e_rbracket))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR080 - Expected ')' after incrementor section of for-loop",
                             exprtk_error_location));

               result = false;
            }
         }

         if (result)
         {
            brkcnt_list_.push_front(false);

            scoped_inc_dec sid(state_.parsing_loop_stmt_count);

            if (0 == (loop_body = parse_multi_sequence("for-loop")))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR081 - Failed to parse body of for-loop",
                             exprtk_error_location));

               result = false;
            }
         }

         if (!result)
         {
            if (se)
            {
               se->ref_count--;
            }

            free_node(node_allocator_, initialiser);
            free_node(node_allocator_, condition  );
            free_node(node_allocator_, incrementor);
            free_node(node_allocator_, loop_body  );

            if (!brkcnt_list_.empty())
            {
               brkcnt_list_.pop_front();
            }

            return error_node();
         }
         else
         {
            expression_node_ptr result_node =
               expression_generator_.for_loop(initialiser,
                                              condition,
                                              incrementor,
                                              loop_body,
                                              brkcnt_list_.front());
            brkcnt_list_.pop_front();

            return result_node;
         }
      }

      inline expression_node_ptr parse_switch_statement()
      {
         std::vector<expression_node_ptr> arg_list;
         expression_node_ptr result = error_node();

         if (!details::imatch(current_token().value,"switch"))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR082 - Expected keyword 'switch'",
                          exprtk_error_location));

            return error_node();
         }

         scoped_vec_delete<expression_node_t> svd((*this),arg_list);

         next_token();

         if (!token_is(token_t::e_lcrlbracket))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR083 - Expected '{' for call to switch statement",
                          exprtk_error_location));

            return error_node();
         }

         expression_node_ptr default_statement = error_node();

         scoped_expression_delete defstmt_delete((*this), default_statement);

         for ( ; ; )
         {
            if (details::imatch("case",current_token().value))
            {
               next_token();

               expression_node_ptr condition = parse_expression();

               if (0 == condition)
                  return error_node();
               else if (!token_is(token_t::e_colon))
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                current_token(),
                                "ERR084 - Expected ':' for case of switch statement",
                                exprtk_error_location));

                  free_node(node_allocator_, condition);

                  return error_node();
               }

               expression_node_ptr consequent = parse_expression();

               if (0 == consequent)
               {
                  free_node(node_allocator_, condition);

                  return error_node();
               }
               else if (!token_is(token_t::e_eof))
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                current_token(),
                                "ERR085 - Expected ';' at end of case for switch statement",
                                exprtk_error_location));

                  free_node(node_allocator_, condition );
                  free_node(node_allocator_, consequent);

                  return error_node();
               }

               // Can we optimise away the case statement?
               if (is_constant_node(condition) && is_false(condition))
               {
                  free_node(node_allocator_, condition );
                  free_node(node_allocator_, consequent);
               }
               else
               {
                  arg_list.push_back(condition );
                  arg_list.push_back(consequent);
               }

            }
            else if (details::imatch("default",current_token().value))
            {
               if (0 != default_statement)
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                current_token(),
                                "ERR086 - Multiple default cases for switch statement",
                                exprtk_error_location));

                  return error_node();
               }

               next_token();

               if (!token_is(token_t::e_colon))
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                current_token(),
                                "ERR087 - Expected ':' for default of switch statement",
                                exprtk_error_location));

                  return error_node();
               }

               if (token_is(token_t::e_lcrlbracket,prsrhlpr_t::e_hold))
                  default_statement = parse_multi_sequence("switch-default");
               else
                  default_statement = parse_expression();

               if (0 == default_statement)
                  return error_node();
               else if (!token_is(token_t::e_eof))
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                current_token(),
                                "ERR088 - Expected ';' at end of default for switch statement",
                                exprtk_error_location));

                  return error_node();
               }
            }
            else if (token_is(token_t::e_rcrlbracket))
               break;
            else
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR089 - Expected '}' at end of switch statement",
                             exprtk_error_location));

               return error_node();
            }
         }

         const bool default_statement_present = (0 != default_statement);

         if (default_statement_present)
         {
            arg_list.push_back(default_statement);
         }

         result = expression_generator_.switch_statement(arg_list, (0 != default_statement));

         svd.delete_ptr = (0 == result);
         defstmt_delete.delete_ptr = (0 == result);

         return result;
      }

      inline expression_node_ptr parse_multi_switch_statement()
      {
         std::vector<expression_node_ptr> arg_list;

         if (!details::imatch(current_token().value,"[*]"))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR090 - Expected token '[*]'",
                          exprtk_error_location));

            return error_node();
         }

         scoped_vec_delete<expression_node_t> svd((*this),arg_list);

         next_token();

         if (!token_is(token_t::e_lcrlbracket))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR091 - Expected '{' for call to [*] statement",
                          exprtk_error_location));

            return error_node();
         }

         for ( ; ; )
         {
            if (!details::imatch("case",current_token().value))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR092 - Expected a 'case' statement for multi-switch",
                             exprtk_error_location));

               return error_node();
            }

            next_token();

            expression_node_ptr condition = parse_expression();

            if (0 == condition)
               return error_node();

            if (!token_is(token_t::e_colon))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR093 - Expected ':' for case of [*] statement",
                             exprtk_error_location));

               return error_node();
            }

            expression_node_ptr consequent = parse_expression();

            if (0 == consequent)
               return error_node();

            if (!token_is(token_t::e_eof))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR094 - Expected ';' at end of case for [*] statement",
                             exprtk_error_location));

               return error_node();
            }

            // Can we optimise away the case statement?
            if (is_constant_node(condition) && is_false(condition))
            {
               free_node(node_allocator_, condition );
               free_node(node_allocator_, consequent);
            }
            else
            {
               arg_list.push_back(condition );
               arg_list.push_back(consequent);
            }

            if (token_is(token_t::e_rcrlbracket,prsrhlpr_t::e_hold))
            {
               break;
            }
         }

         if (!token_is(token_t::e_rcrlbracket))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR095 - Expected '}' at end of [*] statement",
                          exprtk_error_location));

            return error_node();
         }

         const expression_node_ptr result = expression_generator_.multi_switch_statement(arg_list);

         svd.delete_ptr = (0 == result);

         return result;
      }

      inline expression_node_ptr parse_vararg_function()
      {
         std::vector<expression_node_ptr> arg_list;

         details::operator_type opt_type = details::e_default;
         const std::string symbol = current_token().value;

         if (details::imatch(symbol,"~"))
         {
            next_token();
            return parse_multi_sequence();
         }
         else if (details::imatch(symbol,"[*]"))
         {
            return parse_multi_switch_statement();
         }
         else if (details::imatch(symbol, "avg" )) opt_type = details::e_avg ;
         else if (details::imatch(symbol, "mand")) opt_type = details::e_mand;
         else if (details::imatch(symbol, "max" )) opt_type = details::e_max ;
         else if (details::imatch(symbol, "min" )) opt_type = details::e_min ;
         else if (details::imatch(symbol, "mor" )) opt_type = details::e_mor ;
         else if (details::imatch(symbol, "mul" )) opt_type = details::e_prod;
         else if (details::imatch(symbol, "sum" )) opt_type = details::e_sum ;
         else
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR096 - Unsupported vararg function: " + symbol,
                          exprtk_error_location));

            return error_node();
         }

         scoped_vec_delete<expression_node_t> sdd((*this),arg_list);

         lodge_symbol(symbol, e_st_function);

         next_token();

         if (!token_is(token_t::e_lbracket))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR097 - Expected '(' for call to vararg function: " + symbol,
                          exprtk_error_location));

            return error_node();
         }

         for ( ; ; )
         {
            expression_node_ptr arg = parse_expression();

            if (0 == arg)
               return error_node();
            else
               arg_list.push_back(arg);

            if (token_is(token_t::e_rbracket))
               break;
            else if (!token_is(token_t::e_comma))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR098 - Expected ',' for call to vararg function: " + symbol,
                             exprtk_error_location));

               return error_node();
            }
         }

         const expression_node_ptr result = expression_generator_.vararg_function(opt_type,arg_list);

         sdd.delete_ptr = (0 == result);
         return result;
      }

      inline expression_node_ptr parse_string_range_statement(expression_node_ptr&)
      {
         return error_node();
      }

      inline void parse_pending_string_rangesize(expression_node_ptr& expression)
      {
         // Allow no more than 100 range calls, eg: s[][][]...[][]
         const std::size_t max_rangesize_parses = 100;

         std::size_t i = 0;

         while
            (
              (0 != expression)                     &&
              (i++ < max_rangesize_parses)          &&
              error_list_.empty()                   &&
              is_generally_string_node(expression)  &&
              token_is(token_t::e_lsqrbracket,prsrhlpr_t::e_hold)
            )
         {
            expression = parse_string_range_statement(expression);
         }
      }

      template <typename Allocator1,
                typename Allocator2,
                template <typename, typename> class Sequence>
      inline expression_node_ptr simplify(Sequence<expression_node_ptr,Allocator1>& expression_list,
                                          Sequence<bool,Allocator2>& side_effect_list,
                                          const bool specialise_on_final_type = false)
      {
         if (expression_list.empty())
            return error_node();
         else if (1 == expression_list.size())
            return expression_list[0];

         Sequence<expression_node_ptr,Allocator1> tmp_expression_list;

         bool return_node_present = false;

         for (std::size_t i = 0; i < (expression_list.size() - 1); ++i)
         {
            if (is_variable_node(expression_list[i]))
               continue;
            else if (
                      is_return_node  (expression_list[i]) ||
                      is_break_node   (expression_list[i]) ||
                      is_continue_node(expression_list[i])
                    )
            {
               tmp_expression_list.push_back(expression_list[i]);

               // Remove all subexpressions after first short-circuit
               // node has been encountered.

               for (std::size_t j = i + 1; j < expression_list.size(); ++j)
               {
                  free_node(node_allocator_,expression_list[j]);
               }

               return_node_present = true;

               break;
            }
            else if (
                      is_constant_node(expression_list[i]) ||
                      is_null_node    (expression_list[i]) ||
                      !side_effect_list[i]
                    )
            {
               free_node(node_allocator_,expression_list[i]);
               continue;
            }
            else
               tmp_expression_list.push_back(expression_list[i]);
         }

         if (!return_node_present)
         {
            tmp_expression_list.push_back(expression_list.back());
         }

         expression_list.swap(tmp_expression_list);

         if (tmp_expression_list.size() > expression_list.size())
         {
            exprtk_debug(("simplify() - Reduced subexpressions from %d to %d\n",
                          static_cast<int>(tmp_expression_list.size()),
                          static_cast<int>(expression_list    .size())));
         }

         if (
              return_node_present     ||
              side_effect_list.back() ||
              (expression_list.size() > 1)
            )
            state_.activate_side_effect("simplify()");

         if (1 == expression_list.size())
            return expression_list[0];
         else if (specialise_on_final_type && is_generally_string_node(expression_list.back()))
            return expression_generator_.vararg_function(details::e_smulti,expression_list);
         else
            return expression_generator_.vararg_function(details::e_multi,expression_list);
      }

      inline expression_node_ptr parse_multi_sequence(const std::string& source = "")
      {
         token_t::token_type close_bracket = token_t::e_rcrlbracket;
         token_t::token_type seperator     = token_t::e_eof;

         if (!token_is(token_t::e_lcrlbracket))
         {
            if (token_is(token_t::e_lbracket))
            {
               close_bracket = token_t::e_rbracket;
               seperator     = token_t::e_comma;
            }
            else
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR101 - Expected '" + token_t::to_str(close_bracket) + "' for call to multi-sequence" +
                             ((!source.empty()) ? std::string(" section of " + source): ""),
                             exprtk_error_location));

               return error_node();
            }
         }
         else if (token_is(token_t::e_rcrlbracket))
         {
            return node_allocator_.allocate<details::null_node<T> >();
         }

         std::vector<expression_node_ptr> arg_list;
         std::vector<bool> side_effect_list;

         expression_node_ptr result = error_node();

         scoped_vec_delete<expression_node_t> sdd((*this),arg_list);

         scope_handler sh(*this);

         scoped_bool_or_restorer sbr(state_.side_effect_present);

         for ( ; ; )
         {
            state_.side_effect_present = false;

            expression_node_ptr arg = parse_expression();

            if (0 == arg)
               return error_node();
            else
            {
               arg_list.push_back(arg);
               side_effect_list.push_back(state_.side_effect_present);
            }

            if (token_is(close_bracket))
               break;

            const bool is_next_close = peek_token_is(close_bracket);

            if (!token_is(seperator) && is_next_close)
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR102 - Expected '" + details::to_str(seperator) + "' for call to multi-sequence section of " + source,
                             exprtk_error_location));

               return error_node();
            }

            if (token_is(close_bracket))
               break;
         }

         result = simplify(arg_list,side_effect_list,source.empty());

         sdd.delete_ptr = (0 == result);
         return result;
      }

      inline bool parse_range(range_t& rp, const bool skip_lsqr = false)
      {
         // Examples of valid ranges:
         // 1. [1:5]     -> 1..5
         // 2. [ :5]     -> 0..5
         // 3. [1: ]     -> 1..end
         // 4. [x:y]     -> x..y where x <= y
         // 5. [x+1:y/2] -> x+1..y/2 where x+1 <= y/2
         // 6. [ :y]     -> 0..y where 0 <= y
         // 7. [x: ]     -> x..end where x <= end

         rp.clear();

         if (!skip_lsqr && !token_is(token_t::e_lsqrbracket))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR103 - Expected '[' for start of range",
                          exprtk_error_location));

            return false;
         }

         if (token_is(token_t::e_colon))
         {
            rp.n0_c.first  = true;
            rp.n0_c.second = 0;
            rp.cache.first = 0;
         }
         else
         {
            expression_node_ptr r0 = parse_expression();

            if (0 == r0)
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR104 - Failed parse begin section of range",
                             exprtk_error_location));

               return false;
            }
            else if (is_constant_node(r0))
            {
               const T r0_value = r0->value();

               if (r0_value >= T(0))
               {
                  rp.n0_c.first  = true;
                  rp.n0_c.second = static_cast<std::size_t>(details::numeric::to_int64(r0_value));
                  rp.cache.first = rp.n0_c.second;
               }

               free_node(node_allocator_,r0);

               if (r0_value < T(0))
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                current_token(),
                                "ERR105 - Range lower bound less than zero! Constraint: r0 >= 0",
                                exprtk_error_location));

                  return false;
               }
            }
            else
            {
               rp.n0_e.first  = true;
               rp.n0_e.second = r0;
            }

            if (!token_is(token_t::e_colon))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR106 - Expected ':' for break  in range",
                             exprtk_error_location));

               rp.free();

               return false;
            }
         }

         if (token_is(token_t::e_rsqrbracket))
         {
            rp.n1_c.first  = true;
            rp.n1_c.second = std::numeric_limits<std::size_t>::max();
         }
         else
         {
            expression_node_ptr r1 = parse_expression();

            if (0 == r1)
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR107 - Failed parse end section of range",
                             exprtk_error_location));

               rp.free();

               return false;
            }
            else if (is_constant_node(r1))
            {
               const T r1_value = r1->value();

               if (r1_value >= T(0))
               {
                  rp.n1_c.first   = true;
                  rp.n1_c.second  = static_cast<std::size_t>(details::numeric::to_int64(r1_value));
                  rp.cache.second = rp.n1_c.second;
               }

               free_node(node_allocator_,r1);

               if (r1_value < T(0))
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                current_token(),
                                "ERR108 - Range upper bound less than zero! Constraint: r1 >= 0",
                                exprtk_error_location));

                  rp.free();

                  return false;
               }
            }
            else
            {
               rp.n1_e.first  = true;
               rp.n1_e.second = r1;
            }

            if (!token_is(token_t::e_rsqrbracket))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR109 - Expected ']' for start of range",
                             exprtk_error_location));

               rp.free();

               return false;
            }
         }

         if (rp.const_range())
         {
            std::size_t r0 = 0;
            std::size_t r1 = 0;

            bool rp_result = false;

            try
            {
               rp_result = rp(r0, r1);
            }
            catch (std::runtime_error&)
            {}

            if (!rp_result || (r0 > r1))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR110 - Invalid range, Constraint: r0 <= r1",
                             exprtk_error_location));

               return false;
            }
         }

         return true;
      }

      inline void lodge_symbol(const std::string& symbol,
                               const symbol_type st)
      {
         dec_.add_symbol(symbol,st);
      }

      inline expression_node_ptr parse_string()
      {
         return error_node();
      }

      inline expression_node_ptr parse_const_string()
      {
         return error_node();
      }

      inline expression_node_ptr parse_vector()
      {
         const std::string symbol = current_token().value;

         vector_holder_ptr vec = vector_holder_ptr(0);

         const scope_element& se = sem_.get_active_element(symbol);

         if (
              !details::imatch(se.name, symbol) ||
              (se.depth > state_.scope_depth)   ||
              (scope_element::e_vector != se.type)
            )
         {
            if (0 == (vec = symtab_store_.get_vector(symbol)))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR113 - Symbol '" + symbol+ " not a vector",
                             exprtk_error_location));

               return error_node();
            }
         }
         else
            vec = se.vec_node;

         expression_node_ptr index_expr = error_node();

         next_token();

         if (!token_is(token_t::e_lsqrbracket))
         {
            return node_allocator_.allocate<vector_node_t>(vec);
         }
         else if (token_is(token_t::e_rsqrbracket))
         {
            return expression_generator_(T(vec->size()));
         }
         else if (0 == (index_expr = parse_expression()))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR114 - Failed to parse index for vector: '" + symbol + "'",
                          exprtk_error_location));

            return error_node();
         }
         else if (!token_is(token_t::e_rsqrbracket))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR115 - Expected ']' for index of vector: '" + symbol + "'",
                          exprtk_error_location));

            free_node(node_allocator_,index_expr);

            return error_node();
         }

         // Perform compile-time range check
         if (details::is_constant_node(index_expr))
         {
            const std::size_t index    = static_cast<std::size_t>(details::numeric::to_int32(index_expr->value()));
            const std::size_t vec_size = vec->size();

            if (index >= vec_size)
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR116 - Index of " + details::to_str(index) + " out of range for "
                             "vector '" + symbol + "' of size " + details::to_str(vec_size),
                             exprtk_error_location));

               free_node(node_allocator_,index_expr);

               return error_node();
            }
         }

         return expression_generator_.vector_element(symbol, vec, index_expr);
      }

      inline expression_node_ptr parse_vararg_function_call(ivararg_function<T>* vararg_function, const std::string& vararg_function_name)
      {
         std::vector<expression_node_ptr> arg_list;

         expression_node_ptr result = error_node();

         scoped_vec_delete<expression_node_t> sdd((*this),arg_list);

         next_token();

         if (token_is(token_t::e_lbracket))
         {
            if (token_is(token_t::e_rbracket))
            {
               if (!vararg_function->allow_zero_parameters())
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                current_token(),
                                "ERR117 - Zero parameter call to vararg function: "
                                + vararg_function_name + " not allowed",
                                exprtk_error_location));

                  return error_node();
               }
            }
            else
            {
               for ( ; ; )
               {
                  expression_node_ptr arg = parse_expression();

                  if (0 == arg)
                     return error_node();
                  else
                     arg_list.push_back(arg);

                  if (token_is(token_t::e_rbracket))
                     break;
                  else if (!token_is(token_t::e_comma))
                  {
                     set_error(
                        make_error(parser_error::e_syntax,
                                   current_token(),
                                   "ERR118 - Expected ',' for call to vararg function: "
                                   + vararg_function_name,
                                   exprtk_error_location));

                     return error_node();
                  }
               }
            }
         }
         else if (!vararg_function->allow_zero_parameters())
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR119 - Zero parameter call to vararg function: "
                          + vararg_function_name + " not allowed",
                          exprtk_error_location));

            return error_node();
         }

         if (arg_list.size() < vararg_function->min_num_args())
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR120 - Invalid number of parameters to call to vararg function: "
                          + vararg_function_name + ", require at least "
                          + details::to_str(static_cast<int>(vararg_function->min_num_args())) + " parameters",
                          exprtk_error_location));

            return error_node();
         }
         else if (arg_list.size() > vararg_function->max_num_args())
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR121 - Invalid number of parameters to call to vararg function: "
                          + vararg_function_name + ", require no more than "
                          + details::to_str(static_cast<int>(vararg_function->max_num_args())) + " parameters",
                          exprtk_error_location));

            return error_node();
         }

         result = expression_generator_.vararg_function_call(vararg_function,arg_list);

         sdd.delete_ptr = (0 == result);

         return result;
      }

      class type_checker
      {
      public:

         enum return_type_t
         {
            e_overload = ' ',
            e_numeric  = 'T',
            e_string   = 'S'
         };

         struct function_prototype_t
         {
             return_type_t return_type;
             std::string   param_seq;
         };

         typedef parser<T> parser_t;
         typedef std::vector<function_prototype_t> function_definition_list_t;

         type_checker(parser_t& p,
                      const std::string& func_name,
                      const std::string& func_prototypes,
                      const return_type_t default_return_type)
         : invalid_state_(true)
         , parser_(p)
         , function_name_(func_name)
         , default_return_type_(default_return_type)
         {
            parse_function_prototypes(func_prototypes);
         }

         bool verify(const std::string& param_seq, std::size_t& pseq_index)
         {
            if (function_definition_list_.empty())
               return true;

            std::vector<std::pair<std::size_t,char> > error_list;

            for (std::size_t i = 0; i < function_definition_list_.size(); ++i)
            {
               details::char_t diff_value = 0;
               std::size_t     diff_index = 0;

               const bool result = details::sequence_match(function_definition_list_[i].param_seq,
                                                           param_seq,
                                                           diff_index, diff_value);

              if (result)
              {
                 pseq_index = i;
                 return true;
              }
              else
                 error_list.push_back(std::make_pair(diff_index, diff_value));
            }

            if (1 == error_list.size())
            {
               parser_.
                  set_error(
                     make_error(parser_error::e_syntax,
                                parser_.current_token(),
                                "ERR122 - Failed parameter type check for function '" + function_name_ + "', "
                                "Expected '" + function_definition_list_[0].param_seq +
                                "' call set: '" + param_seq + "'",
                                exprtk_error_location));
            }
            else
            {
               // find first with largest diff_index;
               std::size_t max_diff_index = 0;

               for (std::size_t i = 1; i < error_list.size(); ++i)
               {
                  if (error_list[i].first > error_list[max_diff_index].first)
                  {
                     max_diff_index = i;
                  }
               }

               parser_.
                  set_error(
                     make_error(parser_error::e_syntax,
                                parser_.current_token(),
                                "ERR123 - Failed parameter type check for function '" + function_name_ + "', "
                                "Best match: '" + function_definition_list_[max_diff_index].param_seq +
                                "' call set: '" + param_seq + "'",
                                exprtk_error_location));
            }

            return false;
         }

         std::size_t paramseq_count() const
         {
            return function_definition_list_.size();
         }

         std::string paramseq(const std::size_t& index) const
         {
            return function_definition_list_[index].param_seq;
         }

         return_type_t return_type(const std::size_t& index) const
         {
            return function_definition_list_[index].return_type;
         }

         bool invalid() const
         {
            return !invalid_state_;
         }

         bool allow_zero_parameters() const
         {

            for (std::size_t i = 0; i < function_definition_list_.size(); ++i)
            {
               if (std::string::npos != function_definition_list_[i].param_seq.find("Z"))
               {
                  return true;
               }
            }

            return false;
         }

      private:

         std::vector<std::string> split_param_seq(const std::string& param_seq, const details::char_t delimiter = '|') const
         {
             std::string::const_iterator current_begin = param_seq.begin();
             std::string::const_iterator iter          = param_seq.begin();

             std::vector<std::string> result;

             while (iter != param_seq.end())
             {
                 if (*iter == delimiter)
                 {
                     result.push_back(std::string(current_begin, iter));
                     current_begin = ++iter;
                 }
                 else
                     ++iter;
             }

             if (current_begin != iter)
             {
                 result.push_back(std::string(current_begin, iter));
             }

             return result;
         }

         inline bool is_valid_token(std::string param_seq,
                                    function_prototype_t& funcproto) const
         {
            // Determine return type
            funcproto.return_type = default_return_type_;

            if (param_seq.size() > 2)
            {
               if (':' == param_seq[1])
               {
                  // Note: Only overloaded igeneric functions can have return
                  // type definitions.
                  if (type_checker::e_overload != default_return_type_)
                     return false;

                  switch (param_seq[0])
                  {
                     case 'T' : funcproto.return_type = type_checker::e_numeric;
                                break;

                     case 'S' : funcproto.return_type = type_checker::e_string;
                                break;

                     default  : return false;
                  }

                  param_seq.erase(0,2);
               }
            }

            if (
                 (std::string::npos != param_seq.find("?*")) ||
                 (std::string::npos != param_seq.find("**"))
               )
            {
               return false;
            }
            else if (
                      (std::string::npos == param_seq.find_first_not_of("STV*?|")) ||
                      ("Z" == param_seq)
                    )
            {
               funcproto.param_seq = param_seq;
               return true;
            }

            return false;
         }

         void parse_function_prototypes(const std::string& func_prototypes)
         {
            if (func_prototypes.empty())
               return;

            std::vector<std::string> param_seq_list = split_param_seq(func_prototypes);

            typedef std::map<std::string,std::size_t> param_seq_map_t;
            param_seq_map_t param_seq_map;

            for (std::size_t i = 0; i < param_seq_list.size(); ++i)
            {
               function_prototype_t func_proto;

               if (!is_valid_token(param_seq_list[i], func_proto))
               {
                  invalid_state_ = false;

                  parser_.
                     set_error(
                        make_error(parser_error::e_syntax,
                                   parser_.current_token(),
                                   "ERR124 - Invalid parameter sequence of '" + param_seq_list[i] +
                                   "' for function: " + function_name_,
                                   exprtk_error_location));
                  return;
               }

               param_seq_map_t::const_iterator seq_itr = param_seq_map.find(param_seq_list[i]);

               if (param_seq_map.end() != seq_itr)
               {
                  invalid_state_ = false;

                  parser_.
                     set_error(
                        make_error(parser_error::e_syntax,
                                   parser_.current_token(),
                                   "ERR125 - Function '" + function_name_ + "' has a parameter sequence conflict between " +
                                   "pseq_idx[" + details::to_str(seq_itr->second) + "] and" +
                                   "pseq_idx[" + details::to_str(i) + "] " +
                                   "param seq: " + param_seq_list[i],
                                   exprtk_error_location));
                  return;
               }

               function_definition_list_.push_back(func_proto);
            }
         }

         type_checker(const type_checker&) exprtk_delete;
         type_checker& operator=(const type_checker&) exprtk_delete;

         bool invalid_state_;
         parser_t& parser_;
         std::string function_name_;
         const return_type_t default_return_type_;
         function_definition_list_t function_definition_list_;
      };

      inline expression_node_ptr parse_generic_function_call(igeneric_function<T>* function, const std::string& function_name)
      {
         std::vector<expression_node_ptr> arg_list;

         scoped_vec_delete<expression_node_t> sdd((*this),arg_list);

         next_token();

         std::string param_type_list;

         type_checker tc((*this), function_name, function->parameter_sequence, type_checker::e_string);

         if (tc.invalid())
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR126 - Type checker instantiation failure for generic function: " + function_name,
                          exprtk_error_location));

            return error_node();
         }

         if (token_is(token_t::e_lbracket))
         {
            if (token_is(token_t::e_rbracket))
            {
               if (
                    !function->allow_zero_parameters() &&
                    !tc       .allow_zero_parameters()
                  )
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                current_token(),
                                "ERR127 - Zero parameter call to generic function: "
                                + function_name + " not allowed",
                                exprtk_error_location));

                  return error_node();
               }
            }
            else
            {
               for ( ; ; )
               {
                  expression_node_ptr arg = parse_expression();

                  if (0 == arg)
                     return error_node();

                  if (is_ivector_node(arg))
                     param_type_list += 'V';
                  else if (is_generally_string_node(arg))
                     param_type_list += 'S';
                  else // Everything else is assumed to be a scalar returning expression
                     param_type_list += 'T';

                  arg_list.push_back(arg);

                  if (token_is(token_t::e_rbracket))
                     break;
                  else if (!token_is(token_t::e_comma))
                  {
                     set_error(
                        make_error(parser_error::e_syntax,
                                   current_token(),
                                   "ERR128 - Expected ',' for call to generic function: " + function_name,
                                   exprtk_error_location));

                     return error_node();
                  }
               }
            }
         }
         else if (
                   !function->parameter_sequence.empty() &&
                   function->allow_zero_parameters    () &&
                   !tc      .allow_zero_parameters    ()
                 )
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR129 - Zero parameter call to generic function: "
                          + function_name + " not allowed",
                          exprtk_error_location));

            return error_node();
         }

         std::size_t param_seq_index = 0;

         if (
              state_.type_check_enabled &&
              !tc.verify(param_type_list, param_seq_index)
            )
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR130 - Invalid input parameter sequence for call to generic function: " + function_name,
                          exprtk_error_location));

            return error_node();
         }

         expression_node_ptr result = error_node();

         if (tc.paramseq_count() <= 1)
            result = expression_generator_
                       .generic_function_call(function, arg_list);
         else
            result = expression_generator_
                       .generic_function_call(function, arg_list, param_seq_index);

         sdd.delete_ptr = (0 == result);

         return result;
      }

      inline bool parse_igeneric_function_params(std::string& param_type_list,
                                                 std::vector<expression_node_ptr>& arg_list,
                                                 const std::string& function_name,
                                                 igeneric_function<T>* function,
                                                 const type_checker& tc)
      {
         if (token_is(token_t::e_lbracket))
         {
            if (token_is(token_t::e_rbracket))
            {
               if (
                    !function->allow_zero_parameters() &&
                    !tc       .allow_zero_parameters()
                  )
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                current_token(),
                                "ERR131 - Zero parameter call to generic function: "
                                + function_name + " not allowed",
                                exprtk_error_location));

                  return false;
               }
            }
            else
            {
               for ( ; ; )
               {
                  expression_node_ptr arg = parse_expression();

                  if (0 == arg)
                     return false;

                  if (is_ivector_node(arg))
                     param_type_list += 'V';
                  else if (is_generally_string_node(arg))
                     param_type_list += 'S';
                  else // Everything else is a scalar returning expression
                     param_type_list += 'T';

                  arg_list.push_back(arg);

                  if (token_is(token_t::e_rbracket))
                     break;
                  else if (!token_is(token_t::e_comma))
                  {
                     set_error(
                        make_error(parser_error::e_syntax,
                                   current_token(),
                                   "ERR132 - Expected ',' for call to string function: " + function_name,
                                   exprtk_error_location));

                     return false;
                  }
               }
            }

            return true;
         }
         else
            return false;
      }

      template <typename Type, std::size_t NumberOfParameters>
      struct parse_special_function_impl
      {
         static inline expression_node_ptr process(parser<Type>& p, const details::operator_type opt_type, const std::string& sf_name)
         {
            expression_node_ptr branch[NumberOfParameters];
            expression_node_ptr result = error_node();

            std::fill_n(branch, NumberOfParameters, reinterpret_cast<expression_node_ptr>(0));

            scoped_delete<expression_node_t,NumberOfParameters> sd(p,branch);

            p.next_token();

            if (!p.token_is(token_t::e_lbracket))
            {
               p.set_error(
                    make_error(parser_error::e_syntax,
                               p.current_token(),
                               "ERR136 - Expected '(' for special function '" + sf_name + "'",
                               exprtk_error_location));

               return error_node();
            }

            for (std::size_t i = 0; i < NumberOfParameters; ++i)
            {
               branch[i] = p.parse_expression();

               if (0 == branch[i])
               {
                  return p.error_node();
               }
               else if (i < (NumberOfParameters - 1))
               {
                  if (!p.token_is(token_t::e_comma))
                  {
                     p.set_error(
                          make_error(parser_error::e_syntax,
                                     p.current_token(),
                                     "ERR137 - Expected ',' before next parameter of special function '" + sf_name + "'",
                                     exprtk_error_location));

                     return p.error_node();
                  }
               }
            }

            if (!p.token_is(token_t::e_rbracket))
            {
               p.set_error(
                    make_error(parser_error::e_syntax,
                               p.current_token(),
                               "ERR138 - Invalid number of parameters for special function '" + sf_name + "'",
                               exprtk_error_location));

               return p.error_node();
            }
            else
               result = p.expression_generator_.special_function(opt_type,branch);

            sd.delete_ptr = (0 == result);

            return result;
         }
      };

      inline expression_node_ptr parse_special_function()
      {
         const std::string sf_name = current_token().value;

         // Expect: $fDD(expr0,expr1,expr2) or $fDD(expr0,expr1,expr2,expr3)
         if (
              !details::is_digit(sf_name[2]) ||
              !details::is_digit(sf_name[3])
            )
         {
            set_error(
               make_error(parser_error::e_token,
                          current_token(),
                          "ERR139 - Invalid special function[1]: " + sf_name,
                          exprtk_error_location));

            return error_node();
         }

         const int id = (sf_name[2] - '0') * 10 +
                        (sf_name[3] - '0');

         if (id >= details::e_sffinal)
         {
            set_error(
               make_error(parser_error::e_token,
                          current_token(),
                          "ERR140 - Invalid special function[2]: " + sf_name,
                          exprtk_error_location));

            return error_node();
         }

         const int sf_3_to_4                   = details::e_sf48;
         const details::operator_type opt_type = details::operator_type(id + 1000);
         const std::size_t NumberOfParameters  = (id < (sf_3_to_4 - 1000)) ? 3U : 4U;

         switch (NumberOfParameters)
         {
            case 3  : return parse_special_function_impl<T,3>::process((*this), opt_type, sf_name);
            case 4  : return parse_special_function_impl<T,4>::process((*this), opt_type, sf_name);
            default : return error_node();
         }
      }

      inline expression_node_ptr parse_null_statement()
      {
         next_token();
         return node_allocator_.allocate<details::null_node<T> >();
      }

      inline expression_node_ptr parse_define_vector_statement(const std::string& vec_name)
      {
         expression_node_ptr size_expr = error_node();

         if (!token_is(token_t::e_lsqrbracket))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR147 - Expected '[' as part of vector size definition",
                          exprtk_error_location));

            return error_node();
         }
         else if (0 == (size_expr = parse_expression()))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR148 - Failed to determine size of vector '" + vec_name + "'",
                          exprtk_error_location));

            return error_node();
         }
         else if (!is_constant_node(size_expr))
         {
            free_node(node_allocator_,size_expr);

            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR149 - Expected a literal number as size of vector '" + vec_name + "'",
                          exprtk_error_location));

            return error_node();
         }

         const T vector_size = size_expr->value();

         free_node(node_allocator_,size_expr);

         const T max_vector_size = T(2000000000.0);

         if (
              (vector_size <= T(0)) ||
              std::not_equal_to<T>()
              (T(0),vector_size - details::numeric::trunc(vector_size)) ||
              (vector_size > max_vector_size)
            )
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR150 - Invalid vector size. Must be an integer in the range [0,2e9], size: " +
                          details::to_str(details::numeric::to_int32(vector_size)),
                          exprtk_error_location));

            return error_node();
         }

         std::vector<expression_node_ptr> vec_initilizer_list;

         scoped_vec_delete<expression_node_t> svd((*this),vec_initilizer_list);

         bool single_value_initialiser = false;
         bool vec_to_vec_initialiser   = false;
         bool null_initialisation      = false;

         if (!token_is(token_t::e_rsqrbracket))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR151 - Expected ']' as part of vector size definition",
                          exprtk_error_location));

            return error_node();
         }
         else if (!token_is(token_t::e_eof))
         {
            if (!token_is(token_t::e_assign))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR152 - Expected ':=' as part of vector definition",
                             exprtk_error_location));

               return error_node();
            }
            else if (token_is(token_t::e_lsqrbracket))
            {
               expression_node_ptr initialiser = parse_expression();

               if (0 == initialiser)
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                current_token(),
                                "ERR153 - Failed to parse single vector initialiser",
                                exprtk_error_location));

                  return error_node();
               }

               vec_initilizer_list.push_back(initialiser);

               if (!token_is(token_t::e_rsqrbracket))
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                current_token(),
                                "ERR154 - Expected ']' to close single value vector initialiser",
                                exprtk_error_location));

                  return error_node();
               }

               single_value_initialiser = true;
            }
            else if (!token_is(token_t::e_lcrlbracket))
            {
               expression_node_ptr initialiser = error_node();

               // Is this a vector to vector assignment and initialisation?
               if (token_t::e_symbol == current_token().type)
               {
                  // Is it a locally defined vector?
                  const scope_element& se = sem_.get_active_element(current_token().value);

                  if (scope_element::e_vector == se.type)
                  {
                     if (0 != (initialiser = parse_expression()))
                        vec_initilizer_list.push_back(initialiser);
                     else
                        return error_node();
                  }
                  // Are we dealing with a user defined vector?
                  else if (symtab_store_.is_vector(current_token().value))
                  {
                     lodge_symbol(current_token().value, e_st_vector);

                     if (0 != (initialiser = parse_expression()))
                        vec_initilizer_list.push_back(initialiser);
                     else
                        return error_node();
                  }
                  // Are we dealing with a null initialisation vector definition?
                  else if (token_is(token_t::e_symbol,"null"))
                     null_initialisation = true;
               }

               if (!null_initialisation)
               {
                  if (0 == initialiser)
                  {
                     set_error(
                        make_error(parser_error::e_syntax,
                                   current_token(),
                                   "ERR155 - Expected '{' as part of vector initialiser list",
                                   exprtk_error_location));

                     return error_node();
                  }
                  else
                     vec_to_vec_initialiser = true;
               }
            }
            else if (!token_is(token_t::e_rcrlbracket))
            {
               for ( ; ; )
               {
                  expression_node_ptr initialiser = parse_expression();

                  if (0 == initialiser)
                  {
                     set_error(
                        make_error(parser_error::e_syntax,
                                   current_token(),
                                   "ERR156 - Expected '{' as part of vector initialiser list",
                                   exprtk_error_location));

                     return error_node();
                  }
                  else
                     vec_initilizer_list.push_back(initialiser);

                  if (token_is(token_t::e_rcrlbracket))
                     break;

                  const bool is_next_close = peek_token_is(token_t::e_rcrlbracket);

                  if (!token_is(token_t::e_comma) && is_next_close)
                  {
                     set_error(
                        make_error(parser_error::e_syntax,
                                   current_token(),
                                   "ERR157 - Expected ',' between vector initialisers",
                                   exprtk_error_location));

                     return error_node();
                  }

                  if (token_is(token_t::e_rcrlbracket))
                     break;
               }
            }

            if (
                 !token_is(token_t::e_rbracket   , prsrhlpr_t::e_hold) &&
                 !token_is(token_t::e_rcrlbracket, prsrhlpr_t::e_hold) &&
                 !token_is(token_t::e_rsqrbracket, prsrhlpr_t::e_hold)
               )
            {
               if (!token_is(token_t::e_eof))
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                current_token(),
                                "ERR158 - Expected ';' at end of vector definition",
                                exprtk_error_location));

                  return error_node();
               }
            }

            if (vec_initilizer_list.size() > vector_size)
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR159 - Initialiser list larger than the number of elements in the vector: '" + vec_name + "'",
                             exprtk_error_location));

               return error_node();
            }
         }

         typename symbol_table_t::vector_holder_ptr vec_holder = typename symbol_table_t::vector_holder_ptr(0);

         const std::size_t vec_size = static_cast<std::size_t>(details::numeric::to_int32(vector_size));

         scope_element& se = sem_.get_element(vec_name);

         if (se.name == vec_name)
         {
            if (se.active)
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR160 - Illegal redefinition of local vector: '" + vec_name + "'",
                             exprtk_error_location));

               return error_node();
            }
            else if (
                      (se.size == vec_size) &&
                      (scope_element::e_vector == se.type)
                    )
            {
               vec_holder = se.vec_node;
               se.active  = true;
               se.depth   = state_.scope_depth;
               se.ref_count++;
            }
         }

         if (0 == vec_holder)
         {
            scope_element nse;
            nse.name      = vec_name;
            nse.active    = true;
            nse.ref_count = 1;
            nse.type      = scope_element::e_vector;
            nse.depth     = state_.scope_depth;
            nse.size      = vec_size;
            nse.data      = new T[vec_size];
            nse.vec_node  = new typename scope_element::vector_holder_t(reinterpret_cast<T*>(nse.data),nse.size);

            if (!sem_.add_element(nse))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR161 - Failed to add new local vector '" + vec_name + "' to SEM",
                             exprtk_error_location));

               sem_.free_element(nse);

               return error_node();
            }

            vec_holder = nse.vec_node;

            exprtk_debug(("parse_define_vector_statement() - INFO - Added new local vector: %s[%d]\n",
                          nse.name.c_str(),
                          static_cast<int>(nse.size)));
         }

         state_.activate_side_effect("parse_define_vector_statement()");

         lodge_symbol(vec_name, e_st_local_vector);

         expression_node_ptr result = error_node();

         if (null_initialisation)
            result = expression_generator_(T(0.0));
         else if (vec_to_vec_initialiser)
         {
            expression_node_ptr vec_node = node_allocator_.allocate<vector_node_t>(vec_holder);

            result = expression_generator_(
                        details::e_assign,
                        vec_node,
                        vec_initilizer_list[0]);
         }
         else
            result = node_allocator_
                        .allocate<details::vector_assignment_node<T> >(
                           (*vec_holder)[0],
                           vec_size,
                           vec_initilizer_list,
                           single_value_initialiser);

         svd.delete_ptr = (0 == result);

         return result;
      }

      inline expression_node_ptr parse_define_string_statement(const std::string&, expression_node_ptr)
      {
         return error_node();
      }

      inline bool local_variable_is_shadowed(const std::string& symbol)
      {
         const scope_element& se = sem_.get_element(symbol);
         return (se.name == symbol) && se.active;
      }

      inline expression_node_ptr parse_define_var_statement()
      {
         if (settings_.vardef_disabled())
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR164 - Illegal variable definition",
                          exprtk_error_location));

            return error_node();
         }
         else if (!details::imatch(current_token().value,"var"))
         {
            return error_node();
         }
         else
            next_token();

         const std::string var_name = current_token().value;

         expression_node_ptr initialisation_expression = error_node();

         if (!token_is(token_t::e_symbol))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR165 - Expected a symbol for variable definition",
                          exprtk_error_location));

            return error_node();
         }
         else if (details::is_reserved_symbol(var_name))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR166 - Illegal redefinition of reserved keyword: '" + var_name + "'",
                          exprtk_error_location));

            return error_node();
         }
         else if (symtab_store_.symbol_exists(var_name))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR167 - Illegal redefinition of variable '" + var_name + "'",
                          exprtk_error_location));

            return error_node();
         }
         else if (local_variable_is_shadowed(var_name))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR168 - Illegal redefinition of local variable: '" + var_name + "'",
                          exprtk_error_location));

            return error_node();
         }
         else if (token_is(token_t::e_lsqrbracket,prsrhlpr_t::e_hold))
         {
            return parse_define_vector_statement(var_name);
         }
         else if (token_is(token_t::e_lcrlbracket,prsrhlpr_t::e_hold))
         {
            return parse_uninitialised_var_statement(var_name);
         }
         else if (token_is(token_t::e_assign))
         {
            if (0 == (initialisation_expression = parse_expression()))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR169 - Failed to parse initialisation expression",
                             exprtk_error_location));

               return error_node();
            }
         }

         if (
              !token_is(token_t::e_rbracket   , prsrhlpr_t::e_hold) &&
              !token_is(token_t::e_rcrlbracket, prsrhlpr_t::e_hold) &&
              !token_is(token_t::e_rsqrbracket, prsrhlpr_t::e_hold)
            )
         {
            if (!token_is(token_t::e_eof,prsrhlpr_t::e_hold))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR170 - Expected ';' after variable definition",
                             exprtk_error_location));

               free_node(node_allocator_,initialisation_expression);

               return error_node();
            }
         }

         if (
              (0 != initialisation_expression) &&
              details::is_generally_string_node(initialisation_expression)
            )
         {
            return parse_define_string_statement(var_name,initialisation_expression);
         }

         expression_node_ptr var_node = reinterpret_cast<expression_node_ptr>(0);

         scope_element& se = sem_.get_element(var_name);

         if (se.name == var_name)
         {
            if (se.active)
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR171 - Illegal redefinition of local variable: '" + var_name + "'",
                             exprtk_error_location));

               free_node(node_allocator_, initialisation_expression);

               return error_node();
            }
            else if (scope_element::e_variable == se.type)
            {
               var_node  = se.var_node;
               se.active = true;
               se.depth  = state_.scope_depth;
               se.ref_count++;
            }
         }

         if (0 == var_node)
         {
            scope_element nse;
            nse.name      = var_name;
            nse.active    = true;
            nse.ref_count = 1;
            nse.type      = scope_element::e_variable;
            nse.depth     = state_.scope_depth;
            nse.data      = new T(T(0));
            nse.var_node  = node_allocator_.allocate<variable_node_t>(*reinterpret_cast<T*>(nse.data));

            if (!sem_.add_element(nse))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR172 - Failed to add new local variable '" + var_name + "' to SEM",
                             exprtk_error_location));

               free_node(node_allocator_, initialisation_expression);

               sem_.free_element(nse);

               return error_node();
            }

            var_node = nse.var_node;

            exprtk_debug(("parse_define_var_statement() - INFO - Added new local variable: %s\n",nse.name.c_str()));
         }

         state_.activate_side_effect("parse_define_var_statement()");

         lodge_symbol(var_name, e_st_local_variable);

         expression_node_ptr branch[2] = {0};

         branch[0] = var_node;
         branch[1] = initialisation_expression ? initialisation_expression : expression_generator_(T(0));

         return expression_generator_(details::e_assign,branch);
      }

      inline expression_node_ptr parse_uninitialised_var_statement(const std::string& var_name)
      {
         if (
              !token_is(token_t::e_lcrlbracket) ||
              !token_is(token_t::e_rcrlbracket)
            )
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR173 - Expected a '{}' for uninitialised var definition",
                          exprtk_error_location));

            return error_node();
         }
         else if (!token_is(token_t::e_eof,prsrhlpr_t::e_hold))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR174 - Expected ';' after uninitialised variable definition",
                          exprtk_error_location));

            return error_node();
         }

         expression_node_ptr var_node = reinterpret_cast<expression_node_ptr>(0);

         scope_element& se = sem_.get_element(var_name);

         if (se.name == var_name)
         {
            if (se.active)
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR175 - Illegal redefinition of local variable: '" + var_name + "'",
                             exprtk_error_location));

               return error_node();
            }
            else if (scope_element::e_variable == se.type)
            {
               var_node  = se.var_node;
               se.active = true;
               se.ref_count++;
            }
         }

         if (0 == var_node)
         {
            scope_element nse;
            nse.name      = var_name;
            nse.active    = true;
            nse.ref_count = 1;
            nse.type      = scope_element::e_variable;
            nse.depth     = state_.scope_depth;
            nse.ip_index  = sem_.next_ip_index();
            nse.data      = new T(T(0));
            nse.var_node  = node_allocator_.allocate<variable_node_t>(*reinterpret_cast<T*>(nse.data));

            if (!sem_.add_element(nse))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR176 - Failed to add new local variable '" + var_name + "' to SEM",
                             exprtk_error_location));

               sem_.free_element(nse);

               return error_node();
            }

            exprtk_debug(("parse_uninitialised_var_statement() - INFO - Added new local variable: %s\n",
                          nse.name.c_str()));
         }

         lodge_symbol(var_name, e_st_local_variable);

         state_.activate_side_effect("parse_uninitialised_var_statement()");

         return expression_generator_(T(0));
      }

      inline expression_node_ptr parse_swap_statement()
      {
         if (!details::imatch(current_token().value,"swap"))
         {
            return error_node();
         }
         else
            next_token();

         if (!token_is(token_t::e_lbracket))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR177 - Expected '(' at start of swap statement",
                          exprtk_error_location));

            return error_node();
         }

         expression_node_ptr variable0 = error_node();
         expression_node_ptr variable1 = error_node();

         bool variable0_generated = false;
         bool variable1_generated = false;

         const std::string var0_name = current_token().value;

         if (!token_is(token_t::e_symbol,prsrhlpr_t::e_hold))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR178 - Expected a symbol for variable or vector element definition",
                          exprtk_error_location));

            return error_node();
         }
         else if (peek_token_is(token_t::e_lsqrbracket))
         {
            if (0 == (variable0 = parse_vector()))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR179 - First parameter to swap is an invalid vector element: '" + var0_name + "'",
                             exprtk_error_location));

               return error_node();
            }

            variable0_generated = true;
         }
         else
         {
            if (symtab_store_.is_variable(var0_name))
            {
               variable0 = symtab_store_.get_variable(var0_name);
            }

            const scope_element& se = sem_.get_element(var0_name);

            if (
                 (se.active)            &&
                 (se.name == var0_name) &&
                 (scope_element::e_variable == se.type)
               )
            {
               variable0 = se.var_node;
            }

            lodge_symbol(var0_name, e_st_variable);

            if (0 == variable0)
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR180 - First parameter to swap is an invalid variable: '" + var0_name + "'",
                             exprtk_error_location));

               return error_node();
            }
            else
               next_token();
         }

         if (!token_is(token_t::e_comma))
         {
            set_error(
                make_error(parser_error::e_syntax,
                           current_token(),
                           "ERR181 - Expected ',' between parameters to swap",
                           exprtk_error_location));

            if (variable0_generated)
            {
               free_node(node_allocator_,variable0);
            }

            return error_node();
         }

         const std::string var1_name = current_token().value;

         if (!token_is(token_t::e_symbol,prsrhlpr_t::e_hold))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR182 - Expected a symbol for variable or vector element definition",
                          exprtk_error_location));

            if (variable0_generated)
            {
               free_node(node_allocator_,variable0);
            }

            return error_node();
         }
         else if (peek_token_is(token_t::e_lsqrbracket))
         {
            if (0 == (variable1 = parse_vector()))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR183 - Second parameter to swap is an invalid vector element: '" + var1_name + "'",
                             exprtk_error_location));

               if (variable0_generated)
               {
                  free_node(node_allocator_,variable0);
               }

               return error_node();
            }

            variable1_generated = true;
         }
         else
         {
            if (symtab_store_.is_variable(var1_name))
            {
               variable1 = symtab_store_.get_variable(var1_name);
            }

            const scope_element& se = sem_.get_element(var1_name);

            if (
                 (se.active) &&
                 (se.name == var1_name) &&
                 (scope_element::e_variable == se.type)
               )
            {
               variable1 = se.var_node;
            }

            lodge_symbol(var1_name, e_st_variable);

            if (0 == variable1)
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR184 - Second parameter to swap is an invalid variable: '" + var1_name + "'",
                             exprtk_error_location));

               if (variable0_generated)
               {
                  free_node(node_allocator_,variable0);
               }

               return error_node();
            }
            else
               next_token();
         }

         if (!token_is(token_t::e_rbracket))
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR185 - Expected ')' at end of swap statement",
                          exprtk_error_location));

            if (variable0_generated)
            {
               free_node(node_allocator_,variable0);
            }

            if (variable1_generated)
            {
               free_node(node_allocator_,variable1);
            }

            return error_node();
         }

         typedef details::variable_node<T>* variable_node_ptr;

         variable_node_ptr v0 = variable_node_ptr(0);
         variable_node_ptr v1 = variable_node_ptr(0);

         expression_node_ptr result = error_node();

         if (
              (0 != (v0 = dynamic_cast<variable_node_ptr>(variable0))) &&
              (0 != (v1 = dynamic_cast<variable_node_ptr>(variable1)))
            )
         {
            result = node_allocator_.allocate<details::swap_node<T> >(v0, v1);

            if (variable0_generated)
            {
               free_node(node_allocator_,variable0);
            }

            if (variable1_generated)
            {
               free_node(node_allocator_,variable1);
            }
         }
         else
            result = node_allocator_.allocate<details::swap_generic_node<T> >
                        (variable0, variable1);

         state_.activate_side_effect("parse_swap_statement()");

         return result;
      }

      inline expression_node_ptr parse_return_statement()
      {
         return error_node();
      }

      inline bool post_variable_process(const std::string& symbol)
      {
         if (
              peek_token_is(token_t::e_lbracket   ) ||
              peek_token_is(token_t::e_lcrlbracket) ||
              peek_token_is(token_t::e_lsqrbracket)
            )
         {
            if (!settings_.commutative_check_enabled())
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR191 - Invalid sequence of variable '"+ symbol + "' and bracket",
                             exprtk_error_location));

               return false;
            }

            lexer().insert_front(token_t::e_mul);
         }

         return true;
      }

      inline bool post_bracket_process(const typename token_t::token_type& token, expression_node_ptr& branch)
      {
         bool implied_mul = false;

         if (details::is_generally_string_node(branch))
            return true;

         const lexer::parser_helper::token_advance_mode hold = prsrhlpr_t::e_hold;

         switch (token)
         {
            case token_t::e_lcrlbracket : implied_mul = token_is(token_t::e_lbracket   ,hold) ||
                                                        token_is(token_t::e_lcrlbracket,hold) ||
                                                        token_is(token_t::e_lsqrbracket,hold) ;
                                          break;

            case token_t::e_lbracket    : implied_mul = token_is(token_t::e_lbracket   ,hold) ||
                                                        token_is(token_t::e_lcrlbracket,hold) ||
                                                        token_is(token_t::e_lsqrbracket,hold) ;
                                          break;

            case token_t::e_lsqrbracket : implied_mul = token_is(token_t::e_lbracket   ,hold) ||
                                                        token_is(token_t::e_lcrlbracket,hold) ||
                                                        token_is(token_t::e_lsqrbracket,hold) ;
                                          break;

            default                     : return true;
         }

         if (implied_mul)
         {
            if (!settings_.commutative_check_enabled())
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR192 - Invalid sequence of brackets",
                             exprtk_error_location));

               return false;
            }
            else if (token_t::e_eof != current_token().type)
            {
               lexer().insert_front(current_token().type);
               lexer().insert_front(token_t::e_mul);
               next_token();
            }
         }

         return true;
      }

      inline expression_node_ptr parse_symtab_symbol()
      {
         const std::string symbol = current_token().value;

         // Are we dealing with a variable or a special constant?
         expression_node_ptr variable = symtab_store_.get_variable(symbol);

         if (variable)
         {
            if (symtab_store_.is_constant_node(symbol))
            {
               variable = expression_generator_(variable->value());
            }

            if (!post_variable_process(symbol))
               return error_node();

            lodge_symbol(symbol, e_st_variable);
            next_token();

            return variable;
         }

         // Are we dealing with a locally defined variable, vector or string?
         if (!sem_.empty())
         {
            scope_element& se = sem_.get_active_element(symbol);

            if (se.active && details::imatch(se.name, symbol))
            {
               if (scope_element::e_variable == se.type)
               {
                  se.active = true;
                  lodge_symbol(symbol, e_st_local_variable);

                  if (!post_variable_process(symbol))
                     return error_node();

                  next_token();

                  return se.var_node;
               }
               else if (scope_element::e_vector == se.type)
               {
                  return parse_vector();
               }
            }
         }

         {
            // Are we dealing with a function?
            ifunction<T>* function = symtab_store_.get_function(symbol);

            if (function)
            {
               lodge_symbol(symbol, e_st_function);

               expression_node_ptr func_node =
                                      parse_function_invocation(function,symbol);

               if (func_node)
                  return func_node;
               else
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                current_token(),
                                "ERR193 - Failed to generate node for function: '" + symbol + "'",
                                exprtk_error_location));

                  return error_node();
               }
            }
         }

         {
            // Are we dealing with a vararg function?
            ivararg_function<T>* vararg_function = symtab_store_.get_vararg_function(symbol);

            if (vararg_function)
            {
               lodge_symbol(symbol, e_st_function);

               expression_node_ptr vararg_func_node =
                                      parse_vararg_function_call(vararg_function, symbol);

               if (vararg_func_node)
                  return vararg_func_node;
               else
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                current_token(),
                                "ERR194 - Failed to generate node for vararg function: '" + symbol + "'",
                                exprtk_error_location));

                  return error_node();
               }
            }
         }

         {
            // Are we dealing with a vararg generic function?
            igeneric_function<T>* generic_function = symtab_store_.get_generic_function(symbol);

            if (generic_function)
            {
               lodge_symbol(symbol, e_st_function);

               expression_node_ptr genericfunc_node =
                                      parse_generic_function_call(generic_function, symbol);

               if (genericfunc_node)
                  return genericfunc_node;
               else
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                current_token(),
                                "ERR195 - Failed to generate node for generic function: '" + symbol + "'",
                                exprtk_error_location));

                  return error_node();
               }
            }
         }

         // Are we dealing with a vector?
         if (symtab_store_.is_vector(symbol))
         {
            lodge_symbol(symbol, e_st_vector);
            return parse_vector();
         }

         if (details::is_reserved_symbol(symbol))
         {
               if (
                    settings_.function_enabled(symbol) ||
                    !details::is_base_function(symbol)
                  )
               {
                  set_error(
                     make_error(parser_error::e_syntax,
                                current_token(),
                                "ERR198 - Invalid use of reserved symbol '" + symbol + "'",
                                exprtk_error_location));

                  return error_node();
               }
         }

         // Should we handle unknown symbols?
         if (resolve_unknown_symbol_ && unknown_symbol_resolver_)
         {
            if (!(settings_.rsrvd_sym_usr_disabled() && details::is_reserved_symbol(symbol)))
            {
               symbol_table_t& symtab = symtab_store_.get_symbol_table();

               std::string error_message;

               if (unknown_symbol_resolver::e_usrmode_default == unknown_symbol_resolver_->mode)
               {
                  T default_value = T(0);

                  typename unknown_symbol_resolver::usr_symbol_type usr_symbol_type = unknown_symbol_resolver::e_usr_unknown_type;

                  if (unknown_symbol_resolver_->process(symbol, usr_symbol_type, default_value, error_message))
                  {
                     bool create_result = false;

                     switch (usr_symbol_type)
                     {
                        case unknown_symbol_resolver::e_usr_variable_type : create_result = symtab.create_variable(symbol, default_value);
                                                                            break;

                        case unknown_symbol_resolver::e_usr_constant_type : create_result = symtab.add_constant(symbol, default_value);
                                                                            break;

                        default                                           : create_result = false;
                     }

                     if (create_result)
                     {
                        expression_node_ptr var = symtab_store_.get_variable(symbol);

                        if (var)
                        {
                           if (symtab_store_.is_constant_node(symbol))
                           {
                              var = expression_generator_(var->value());
                           }

                           lodge_symbol(symbol, e_st_variable);

                           if (!post_variable_process(symbol))
                              return error_node();

                           next_token();

                           return var;
                        }
                     }
                  }

                  set_error(
                     make_error(parser_error::e_symtab,
                                current_token(),
                                "ERR199 - Failed to create variable: '" + symbol + "'" +
                                (error_message.empty() ? "" : " - " + error_message),
                                exprtk_error_location));

               }
               else if (unknown_symbol_resolver::e_usrmode_extended == unknown_symbol_resolver_->mode)
               {
                  if (unknown_symbol_resolver_->process(symbol, symtab, error_message))
                  {
                     expression_node_ptr result = parse_symtab_symbol();

                     if (result)
                     {
                        return result;
                     }
                  }

                  set_error(
                     make_error(parser_error::e_symtab,
                                current_token(),
                                "ERR200 - Failed to resolve symbol: '" + symbol + "'" +
                                (error_message.empty() ? "" : " - " + error_message),
                                exprtk_error_location));
               }

               return error_node();
            }
         }

         set_error(
            make_error(parser_error::e_syntax,
                       current_token(),
                       "ERR201 - Undefined symbol: '" + symbol + "'",
                       exprtk_error_location));

         return error_node();
      }

      inline expression_node_ptr parse_symbol()
      {
         static const std::string symbol_if       = "if"      ;
         static const std::string symbol_while    = "while"   ;
         static const std::string symbol_repeat   = "repeat"  ;
         static const std::string symbol_for      = "for"     ;
         static const std::string symbol_switch   = "switch"  ;
         static const std::string symbol_null     = "null"    ;
         static const std::string symbol_break    = "break"   ;
         static const std::string symbol_continue = "continue";
         static const std::string symbol_var      = "var"     ;
         static const std::string symbol_swap     = "swap"    ;
         static const std::string symbol_return   = "return"  ;
         static const std::string symbol_not      = "not"     ;

         const std::string symbol = current_token().value;

         if (valid_vararg_operation(symbol))
         {
            return parse_vararg_function();
         }
         else if (details::imatch(symbol, symbol_not))
         {
            return parse_not_statement();
         }
         else if (valid_base_operation(symbol))
         {
            return parse_base_operation();
         }
         else if (
                   details::imatch(symbol, symbol_if) &&
                   settings_.control_struct_enabled(symbol)
                 )
         {
            return parse_conditional_statement();
         }
         else if (
                   details::imatch(symbol, symbol_while) &&
                   settings_.control_struct_enabled(symbol)
                 )
         {
            return parse_while_loop();
         }
         else if (
                   details::imatch(symbol, symbol_repeat) &&
                   settings_.control_struct_enabled(symbol)
                 )
         {
            return parse_repeat_until_loop();
         }
         else if (
                   details::imatch(symbol, symbol_for) &&
                   settings_.control_struct_enabled(symbol)
                 )
         {
            return parse_for_loop();
         }
         else if (
                   details::imatch(symbol, symbol_switch) &&
                   settings_.control_struct_enabled(symbol)
                 )
         {
            return parse_switch_statement();
         }
         else if (details::is_valid_sf_symbol(symbol))
         {
            return parse_special_function();
         }
         else if (details::imatch(symbol, symbol_null))
         {
            return parse_null_statement();
         }
         else if (details::imatch(symbol, symbol_var))
         {
            return parse_define_var_statement();
         }
         else if (details::imatch(symbol, symbol_swap))
         {
            return parse_swap_statement();
         }
         else if (symtab_store_.valid() || !sem_.empty())
         {
            return parse_symtab_symbol();
         }
         else
         {
            set_error(
               make_error(parser_error::e_symtab,
                          current_token(),
                          "ERR202 - Variable or function detected, yet symbol-table is invalid, Symbol: " + symbol,
                          exprtk_error_location));

            return error_node();
         }
      }

      inline expression_node_ptr parse_branch(precedence_level precedence = e_level00)
      {
         stack_limit_handler slh(*this);

         if (!slh)
         {
            return error_node();
         }

         expression_node_ptr branch = error_node();

         if (token_t::e_number == current_token().type)
         {
            T numeric_value = T(0);

            if (details::string_to_real(current_token().value, numeric_value))
            {
               expression_node_ptr literal_exp = expression_generator_(numeric_value);

               if (0 == literal_exp)
               {
                  set_error(
                     make_error(parser_error::e_numeric,
                                current_token(),
                                "ERR203 - Failed generate node for scalar: '" + current_token().value + "'",
                                exprtk_error_location));

                  return error_node();
               }

               next_token();
               branch = literal_exp;
            }
            else
            {
               set_error(
                  make_error(parser_error::e_numeric,
                             current_token(),
                             "ERR204 - Failed to convert '" + current_token().value + "' to a number",
                             exprtk_error_location));

               return error_node();
            }
         }
         else if (token_t::e_symbol == current_token().type)
         {
            branch = parse_symbol();
         }
         else if (token_t::e_lbracket == current_token().type)
         {
            next_token();

            if (0 == (branch = parse_expression()))
               return error_node();
            else if (!token_is(token_t::e_rbracket))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR205 - Expected ')' instead of: '" + current_token().value + "'",
                             exprtk_error_location));

               details::free_node(node_allocator_,branch);

               return error_node();
            }
            else if (!post_bracket_process(token_t::e_lbracket,branch))
            {
               details::free_node(node_allocator_,branch);

               return error_node();
            }
         }
         else if (token_t::e_lsqrbracket == current_token().type)
         {
            next_token();

            if (0 == (branch = parse_expression()))
               return error_node();
            else if (!token_is(token_t::e_rsqrbracket))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR206 - Expected ']' instead of: '" + current_token().value + "'",
                             exprtk_error_location));

               details::free_node(node_allocator_,branch);

               return error_node();
            }
            else if (!post_bracket_process(token_t::e_lsqrbracket,branch))
            {
               details::free_node(node_allocator_,branch);

               return error_node();
            }
         }
         else if (token_t::e_lcrlbracket == current_token().type)
         {
            next_token();

            if (0 == (branch = parse_expression()))
               return error_node();
            else if (!token_is(token_t::e_rcrlbracket))
            {
               set_error(
                  make_error(parser_error::e_syntax,
                             current_token(),
                             "ERR207 - Expected '}' instead of: '" + current_token().value + "'",
                             exprtk_error_location));

               details::free_node(node_allocator_,branch);

               return error_node();
            }
            else if (!post_bracket_process(token_t::e_lcrlbracket,branch))
            {
               details::free_node(node_allocator_,branch);

               return error_node();
            }
         }
         else if (token_t::e_sub == current_token().type)
         {
            next_token();
            branch = parse_expression(e_level11);

            if (
                 branch &&
                 !(
                    details::is_neg_unary_node    (branch) &&
                    simplify_unary_negation_branch(branch)
                  )
               )
            {
               expression_node_ptr result = expression_generator_(details::e_neg,branch);

               if (0 == result)
               {
                  details::free_node(node_allocator_,branch);

                  return error_node();
               }
               else
                  branch = result;
            }
         }
         else if (token_t::e_add == current_token().type)
         {
            next_token();
            branch = parse_expression(e_level13);
         }
         else if (token_t::e_eof == current_token().type)
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR208 - Premature end of expression[1]",
                          exprtk_error_location));

            return error_node();
         }
         else
         {
            set_error(
               make_error(parser_error::e_syntax,
                          current_token(),
                          "ERR209 - Premature end of expression[2]",
                          exprtk_error_location));

            return error_node();
         }

         if (
              branch                    &&
              (e_level00 == precedence) &&
              token_is(token_t::e_ternary,prsrhlpr_t::e_hold)
            )
         {
            branch = parse_ternary_conditional_statement(branch);
         }

         parse_pending_string_rangesize(branch);

         return branch;
      }

      template <typename Type>
      class expression_generator
      {
      public:

         typedef details::expression_node<Type>* expression_node_ptr;
         typedef expression_node_ptr (*synthesize_functor_t)(expression_generator<T>&, const details::operator_type& operation, expression_node_ptr (&branch)[2]);
         typedef std::map<std::string,synthesize_functor_t> synthesize_map_t;
         typedef typename exprtk::parser<Type> parser_t;
         typedef const Type& vtype;
         typedef const Type  ctype;

         inline void init_synthesize_map()
         {
         }

         inline void set_parser(parser_t& p)
         {
            parser_ = &p;
         }

         inline void set_uom(unary_op_map_t& unary_op_map)
         {
            unary_op_map_ = &unary_op_map;
         }

         inline void set_bom(binary_op_map_t& binary_op_map)
         {
            binary_op_map_ = &binary_op_map;
         }

         inline void set_ibom(inv_binary_op_map_t& inv_binary_op_map)
         {
            inv_binary_op_map_ = &inv_binary_op_map;
         }

         inline void set_sf3m(sf3_map_t& sf3_map)
         {
            sf3_map_ = &sf3_map;
         }

         inline void set_sf4m(sf4_map_t& sf4_map)
         {
            sf4_map_ = &sf4_map;
         }

         inline void set_allocator(details::node_allocator& na)
         {
            node_allocator_ = &na;
         }

         inline void set_strength_reduction_state(const bool enabled)
         {
            strength_reduction_enabled_ = enabled;
         }

         inline bool strength_reduction_enabled() const
         {
            return strength_reduction_enabled_;
         }

         inline bool valid_operator(const details::operator_type& operation, binary_functor_t& bop)
         {
            typename binary_op_map_t::iterator bop_itr = binary_op_map_->find(operation);

            if ((*binary_op_map_).end() == bop_itr)
               return false;

            bop = bop_itr->second;

            return true;
         }

         inline bool valid_operator(const details::operator_type& operation, unary_functor_t& uop)
         {
            typename unary_op_map_t::iterator uop_itr = unary_op_map_->find(operation);

            if ((*unary_op_map_).end() == uop_itr)
               return false;

            uop = uop_itr->second;

            return true;
         }

         inline details::operator_type get_operator(const binary_functor_t& bop) const
         {
            return (*inv_binary_op_map_).find(bop)->second;
         }

         inline expression_node_ptr operator() (const Type& v) const
         {
            return node_allocator_->allocate<literal_node_t>(v);
         }

         inline bool unary_optimisable(const details::operator_type& operation) const
         {
            return (details::e_abs   == operation) || (details::e_acos  == operation) ||
                   (details::e_acosh == operation) || (details::e_asin  == operation) ||
                   (details::e_asinh == operation) || (details::e_atan  == operation) ||
                   (details::e_atanh == operation) || (details::e_ceil  == operation) ||
                   (details::e_cos   == operation) || (details::e_cosh  == operation) ||
                   (details::e_exp   == operation) || (details::e_expm1 == operation) ||
                   (details::e_floor == operation) || (details::e_log   == operation) ||
                   (details::e_log10 == operation) || (details::e_log2  == operation) ||
                   (details::e_log1p == operation) || (details::e_neg   == operation) ||
                   (details::e_pos   == operation) || (details::e_round == operation) ||
                   (details::e_sin   == operation) || (details::e_sinc  == operation) ||
                   (details::e_sinh  == operation) || (details::e_sqrt  == operation) ||
                   (details::e_tan   == operation) || (details::e_tanh  == operation) ||
                   (details::e_cot   == operation) || (details::e_sec   == operation) ||
                   (details::e_csc   == operation) || (details::e_r2d   == operation) ||
                   (details::e_d2r   == operation) || (details::e_d2g   == operation) ||
                   (details::e_g2d   == operation) || (details::e_notl  == operation) ||
                   (details::e_sgn   == operation) || (details::e_erf   == operation) ||
                   (details::e_erfc  == operation) || (details::e_ncdf  == operation) ||
                   (details::e_frac  == operation) || (details::e_trunc == operation) ;
         }

         inline bool sf3_optimisable(const std::string& sf3id, trinary_functor_t& tfunc) const
         {
            typename sf3_map_t::const_iterator itr = sf3_map_->find(sf3id);

            if (sf3_map_->end() == itr)
               return false;
            else
               tfunc = itr->second.first;

            return true;
         }

         inline bool sf4_optimisable(const std::string& sf4id, quaternary_functor_t& qfunc) const
         {
            typename sf4_map_t::const_iterator itr = sf4_map_->find(sf4id);

            if (sf4_map_->end() == itr)
               return false;
            else
               qfunc = itr->second.first;

            return true;
         }

         inline bool sf3_optimisable(const std::string& sf3id, details::operator_type& operation) const
         {
            typename sf3_map_t::const_iterator itr = sf3_map_->find(sf3id);

            if (sf3_map_->end() == itr)
               return false;
            else
               operation = itr->second.second;

            return true;
         }

         inline bool sf4_optimisable(const std::string& sf4id, details::operator_type& operation) const
         {
            typename sf4_map_t::const_iterator itr = sf4_map_->find(sf4id);

            if (sf4_map_->end() == itr)
               return false;
            else
               operation = itr->second.second;

            return true;
         }

         inline expression_node_ptr operator() (const details::operator_type& operation, expression_node_ptr (&branch)[1])
         {
            if (0 == branch[0])
            {
               return error_node();
            }
            else if (details::is_null_node(branch[0]))
            {
               return branch[0];
            }
            else if (details::is_break_node(branch[0]))
            {
               return error_node();
            }
            else if (details::is_continue_node(branch[0]))
            {
               return error_node();
            }
            else if (details::is_constant_node(branch[0]))
            {
               return synthesize_expression<unary_node_t,1>(operation,branch);
            }
            else if (unary_optimisable(operation) && details::is_variable_node(branch[0]))
            {
               return synthesize_uv_expression(operation,branch);
            }
            else if (unary_optimisable(operation) && details::is_ivector_node(branch[0]))
            {
               return synthesize_uvec_expression(operation,branch);
            }
            else
               return synthesize_unary_expression(operation,branch);
         }

         inline bool is_assignment_operation(const details::operator_type& operation) const
         {
            return (
                     (details::e_addass == operation) ||
                     (details::e_subass == operation) ||
                     (details::e_mulass == operation) ||
                     (details::e_divass == operation) ||
                     (details::e_modass == operation)
                   ) &&
                   parser_->settings_.assignment_enabled(operation);
         }

         inline bool valid_string_operation(const details::operator_type&) const
         {
            return false;
         }

         inline std::string to_str(const details::operator_type& operation) const
         {
            switch (operation)
            {
               case details::e_add  : return "+"      ;
               case details::e_sub  : return "-"      ;
               case details::e_mul  : return "*"      ;
               case details::e_div  : return "/"      ;
               case details::e_mod  : return "%"      ;
               case details::e_pow  : return "^"      ;
               case details::e_lt   : return "<"      ;
               case details::e_lte  : return "<="     ;
               case details::e_gt   : return ">"      ;
               case details::e_gte  : return ">="     ;
               case details::e_eq   : return "=="     ;
               case details::e_ne   : return "!="     ;
               case details::e_and  : return "and"    ;
               case details::e_nand : return "nand"   ;
               case details::e_or   : return "or"     ;
               case details::e_nor  : return "nor"    ;
               case details::e_xor  : return "xor"    ;
               case details::e_xnor : return "xnor"   ;
               default              : return "UNKNOWN";
            }
         }

         inline bool operation_optimisable(const details::operator_type& operation) const
         {
            return (details::e_add  == operation) ||
                   (details::e_sub  == operation) ||
                   (details::e_mul  == operation) ||
                   (details::e_div  == operation) ||
                   (details::e_mod  == operation) ||
                   (details::e_pow  == operation) ||
                   (details::e_lt   == operation) ||
                   (details::e_lte  == operation) ||
                   (details::e_gt   == operation) ||
                   (details::e_gte  == operation) ||
                   (details::e_eq   == operation) ||
                   (details::e_ne   == operation) ||
                   (details::e_and  == operation) ||
                   (details::e_nand == operation) ||
                   (details::e_or   == operation) ||
                   (details::e_nor  == operation) ||
                   (details::e_xor  == operation) ||
                   (details::e_xnor == operation) ;
         }

         inline std::string branch_to_id(expression_node_ptr branch) const
         {
            static const std::string null_str   ("(null)" );
            static const std::string const_str  ("(c)"    );
            static const std::string var_str    ("(v)"    );
            static const std::string vov_str    ("(vov)"  );
            static const std::string cov_str    ("(cov)"  );
            static const std::string voc_str    ("(voc)"  );
            static const std::string str_str    ("(s)"    );
            static const std::string strrng_str ("(rngs)" );
            static const std::string cs_str     ("(cs)"   );
            static const std::string cstrrng_str("(crngs)");

            if (details::is_null_node(branch))
               return null_str;
            else if (details::is_constant_node(branch))
               return const_str;
            else if (details::is_variable_node(branch))
               return var_str;
            else if (details::is_vov_node(branch))
               return vov_str;
            else if (details::is_cov_node(branch))
               return cov_str;
            else if (details::is_voc_node(branch))
               return voc_str;
            else if (details::is_string_node(branch))
               return str_str;
            else if (details::is_const_string_node(branch))
               return cs_str;
            else if (details::is_string_range_node(branch))
               return strrng_str;
            else if (details::is_const_string_range_node(branch))
               return cstrrng_str;
            else if (details::is_t0ot1ot2_node(branch))
               return "(" + dynamic_cast<details::T0oT1oT2_base_node<T>*>(branch)->type_id() + ")";
            else if (details::is_t0ot1ot2ot3_node(branch))
               return "(" + dynamic_cast<details::T0oT1oT2oT3_base_node<T>*>(branch)->type_id() + ")";
            else
               return "ERROR";
         }

         inline std::string branch_to_id(expression_node_ptr (&branch)[2]) const
         {
            return branch_to_id(branch[0]) + std::string("o") + branch_to_id(branch[1]);
         }

         inline bool cov_optimisable(const details::operator_type& operation, expression_node_ptr (&branch)[2]) const
         {
            if (!operation_optimisable(operation))
               return false;
            else
               return details::is_constant_node(branch[0]) &&
                      details::is_variable_node(branch[1]) ;
         }

         inline bool voc_optimisable(const details::operator_type& operation, expression_node_ptr (&branch)[2]) const
         {
            if (!operation_optimisable(operation))
               return false;
            else
               return details::is_variable_node(branch[0]) &&
                      details::is_constant_node(branch[1]) ;
         }

         inline bool vov_optimisable(const details::operator_type& operation, expression_node_ptr (&branch)[2]) const
         {
            if (!operation_optimisable(operation))
               return false;
            else
               return details::is_variable_node(branch[0]) &&
                      details::is_variable_node(branch[1]) ;
         }

         inline bool cob_optimisable(const details::operator_type& operation, expression_node_ptr (&branch)[2]) const
         {
            if (!operation_optimisable(operation))
               return false;
            else
               return details::is_constant_node(branch[0]) &&
                     !details::is_constant_node(branch[1]) ;
         }

         inline bool boc_optimisable(const details::operator_type& operation, expression_node_ptr (&branch)[2]) const
         {
            if (!operation_optimisable(operation))
               return false;
            else
               return !details::is_constant_node(branch[0]) &&
                       details::is_constant_node(branch[1]) ;
         }

         inline bool cocob_optimisable(const details::operator_type& operation, expression_node_ptr (&branch)[2]) const
         {
            if (
                 (details::e_add == operation) ||
                 (details::e_sub == operation) ||
                 (details::e_mul == operation) ||
                 (details::e_div == operation)
               )
            {
               return (details::is_constant_node(branch[0]) && details::is_cob_node(branch[1])) ||
                      (details::is_constant_node(branch[1]) && details::is_cob_node(branch[0])) ;
            }
            else
               return false;
         }

         inline bool coboc_optimisable(const details::operator_type& operation, expression_node_ptr (&branch)[2]) const
         {
            if (
                 (details::e_add == operation) ||
                 (details::e_sub == operation) ||
                 (details::e_mul == operation) ||
                 (details::e_div == operation)
               )
            {
               return (details::is_constant_node(branch[0]) && details::is_boc_node(branch[1])) ||
                      (details::is_constant_node(branch[1]) && details::is_boc_node(branch[0])) ;
            }
            else
               return false;
         }

         inline bool uvouv_optimisable(const details::operator_type& operation, expression_node_ptr (&branch)[2]) const
         {
            if (!operation_optimisable(operation))
               return false;
            else
               return details::is_uv_node(branch[0]) &&
                      details::is_uv_node(branch[1]) ;
         }

         inline bool vob_optimisable(const details::operator_type& operation, expression_node_ptr (&branch)[2]) const
         {
            if (!operation_optimisable(operation))
               return false;
            else
               return details::is_variable_node(branch[0]) &&
                     !details::is_variable_node(branch[1]) ;
         }

         inline bool bov_optimisable(const details::operator_type& operation, expression_node_ptr (&branch)[2]) const
         {
            if (!operation_optimisable(operation))
               return false;
            else
               return !details::is_variable_node(branch[0]) &&
                       details::is_variable_node(branch[1]) ;
         }

         inline bool binext_optimisable(const details::operator_type& operation, expression_node_ptr (&branch)[2]) const
         {
            if (!operation_optimisable(operation))
               return false;
            else
               return !details::is_constant_node(branch[0]) ||
                      !details::is_constant_node(branch[1]) ;
         }

         inline bool is_invalid_assignment_op(const details::operator_type& operation, expression_node_ptr (&branch)[2]) const
         {
            if (is_assignment_operation(operation))
            {
               const bool b1_is_genstring = details::is_generally_string_node(branch[1]);

               if (details::is_string_node(branch[0]))
                  return !b1_is_genstring;
               else
                  return (
                           !details::is_variable_node          (branch[0]) &&
                           !details::is_vector_elem_node       (branch[0]) &&
                           !details::is_rebasevector_elem_node (branch[0]) &&
                           !details::is_rebasevector_celem_node(branch[0]) &&
                           !details::is_vector_node            (branch[0])
                         )
                         || b1_is_genstring;
            }
            else
               return false;
         }

         inline bool is_constpow_operation(const details::operator_type& operation, expression_node_ptr(&branch)[2]) const
         {
            if (
                 !details::is_constant_node(branch[1]) ||
                  details::is_constant_node(branch[0]) ||
                  details::is_variable_node(branch[0]) ||
                  details::is_vector_node  (branch[0]) ||
                  details::is_generally_string_node(branch[0])
               )
               return false;

            const Type c = static_cast<details::literal_node<Type>*>(branch[1])->value();

            return cardinal_pow_optimisable(operation, c);
         }

         inline bool is_invalid_break_continue_op(expression_node_ptr (&branch)[2]) const
         {
            return (
                     details::is_break_node   (branch[0]) ||
                     details::is_break_node   (branch[1]) ||
                     details::is_continue_node(branch[0]) ||
                     details::is_continue_node(branch[1])
                   );
         }

         inline bool is_invalid_string_op(const details::operator_type& operation, expression_node_ptr (&branch)[2]) const
         {
            const bool b0_string = is_generally_string_node(branch[0]);
            const bool b1_string = is_generally_string_node(branch[1]);

            bool result = false;

            if (b0_string != b1_string)
               result = true;
            else if (!valid_string_operation(operation) && b0_string && b1_string)
               result = true;

            if (result)
            {
               parser_->set_synthesis_error("Invalid string operation");
            }

            return result;
         }

         inline bool is_invalid_string_op(const details::operator_type& operation, expression_node_ptr (&branch)[3]) const
         {
            const bool b0_string = is_generally_string_node(branch[0]);
            const bool b1_string = is_generally_string_node(branch[1]);
            const bool b2_string = is_generally_string_node(branch[2]);

            bool result = false;

            if ((b0_string != b1_string) || (b1_string != b2_string))
               result = true;
            else if ((details::e_inrange != operation) && b0_string && b1_string && b2_string)
               result = true;

            if (result)
            {
               parser_->set_synthesis_error("Invalid string operation");
            }

            return result;
         }

         inline bool is_string_operation(const details::operator_type& operation, expression_node_ptr (&branch)[2]) const
         {
            const bool b0_string = is_generally_string_node(branch[0]);
            const bool b1_string = is_generally_string_node(branch[1]);

            return (b0_string && b1_string && valid_string_operation(operation));
         }

         inline bool is_string_operation(const details::operator_type& operation, expression_node_ptr (&branch)[3]) const
         {
            const bool b0_string = is_generally_string_node(branch[0]);
            const bool b1_string = is_generally_string_node(branch[1]);
            const bool b2_string = is_generally_string_node(branch[2]);

            return (b0_string && b1_string && b2_string && (details::e_inrange == operation));
         }

         inline bool is_shortcircuit_expression(const details::operator_type&) const
         {
            return false;
         }

         inline bool is_null_present(expression_node_ptr (&branch)[2]) const
         {
            return (
                     details::is_null_node(branch[0]) ||
                     details::is_null_node(branch[1])
                   );
         }

         inline bool is_vector_eqineq_logic_operation(const details::operator_type& operation, expression_node_ptr (&branch)[2]) const
         {
            if (!is_ivector_node(branch[0]) && !is_ivector_node(branch[1]))
               return false;
            else
               return (
                        (details::e_lt    == operation) ||
                        (details::e_lte   == operation) ||
                        (details::e_gt    == operation) ||
                        (details::e_gte   == operation) ||
                        (details::e_eq    == operation) ||
                        (details::e_ne    == operation) ||
                        (details::e_equal == operation) ||
                        (details::e_and   == operation) ||
                        (details::e_nand  == operation) ||
                        (details::e_or    == operation) ||
                        (details::e_nor   == operation) ||
                        (details::e_xor   == operation) ||
                        (details::e_xnor  == operation)
                      );
         }

         inline bool is_vector_arithmetic_operation(const details::operator_type& operation, expression_node_ptr (&branch)[2]) const
         {
            if (!is_ivector_node(branch[0]) && !is_ivector_node(branch[1]))
               return false;
            else
               return (
                        (details::e_add == operation) ||
                        (details::e_sub == operation) ||
                        (details::e_mul == operation) ||
                        (details::e_div == operation) ||
                        (details::e_pow == operation)
                      );
         }

         inline expression_node_ptr operator() (const details::operator_type& operation, expression_node_ptr (&branch)[2])
         {
            if ((0 == branch[0]) || (0 == branch[1]))
            {
               return error_node();
            }
            else if (is_invalid_string_op(operation,branch))
            {
               return error_node();
            }
            else if (is_invalid_assignment_op(operation,branch))
            {
               return error_node();
            }
            else if (is_invalid_break_continue_op(branch))
            {
               return error_node();
            }
            else if (details::e_assign == operation)
            {
               return synthesize_assignment_expression(operation, branch);
            }
            else if (details::e_swap == operation)
            {
               return synthesize_swap_expression(branch);
            }
            else if (is_assignment_operation(operation))
            {
               return synthesize_assignment_operation_expression(operation, branch);
            }
            else if (is_vector_eqineq_logic_operation(operation, branch))
            {
               return synthesize_veceqineqlogic_operation_expression(operation, branch);
            }
            else if (is_vector_arithmetic_operation(operation, branch))
            {
               return synthesize_vecarithmetic_operation_expression(operation, branch);
            }
            else if (is_shortcircuit_expression(operation))
            {
               return synthesize_shortcircuit_expression(operation, branch);
            }
            else if (is_string_operation(operation, branch))
            {
               return synthesize_string_expression(operation, branch);
            }
            else if (is_null_present(branch))
            {
               return synthesize_null_expression(operation, branch);
            }
            #ifndef exprtk_disable_cardinal_pow_optimisation
            else if (is_constpow_operation(operation, branch))
            {
               return cardinal_pow_optimisation(branch);
            }
            #endif

            expression_node_ptr result = error_node();

            {
               /*
                  Possible reductions:
                  1. c o cob -> cob
                  2. cob o c -> cob
                  3. c o boc -> boc
                  4. boc o c -> boc
               */
               result = error_node();

               if (cocob_optimisable(operation, branch))
               {
                  result = synthesize_cocob_expression::process((*this), operation, branch);
               }
               else if (coboc_optimisable(operation, branch) && (0 == result))
               {
                  result = synthesize_coboc_expression::process((*this), operation, branch);
               }

               if (result)
                  return result;
            }

            if (uvouv_optimisable(operation, branch))
            {
               return synthesize_uvouv_expression(operation, branch);
            }
            else if (vob_optimisable(operation, branch))
            {
               return synthesize_vob_expression::process((*this), operation, branch);
            }
            else if (bov_optimisable(operation, branch))
            {
               return synthesize_bov_expression::process((*this), operation, branch);
            }
            else if (cob_optimisable(operation, branch))
            {
               return synthesize_cob_expression::process((*this), operation, branch);
            }
            else if (boc_optimisable(operation, branch))
            {
               return synthesize_boc_expression::process((*this), operation, branch);
            }
            else if (binext_optimisable(operation, branch))
            {
               return synthesize_binary_ext_expression::process((*this), operation, branch);
            }
            else
               return synthesize_expression<binary_node_t,2>(operation, branch);
         }

         inline expression_node_ptr operator() (const details::operator_type& operation, expression_node_ptr (&branch)[3])
         {
            if (
                 (0 == branch[0]) ||
                 (0 == branch[1]) ||
                 (0 == branch[2])
               )
            {
               details::free_all_nodes(*node_allocator_,branch);

               return error_node();
            }
            else if (is_invalid_string_op(operation, branch))
            {
               return error_node();
            }
            else if (is_string_operation(operation, branch))
            {
               return synthesize_string_expression(operation, branch);
            }
            else
               return synthesize_expression<trinary_node_t,3>(operation, branch);
         }

         inline expression_node_ptr operator() (const details::operator_type& operation, expression_node_ptr (&branch)[4])
         {
            return synthesize_expression<quaternary_node_t,4>(operation,branch);
         }

         inline expression_node_ptr operator() (const details::operator_type& operation, expression_node_ptr b0)
         {
            expression_node_ptr branch[1] = { b0 };
            return (*this)(operation,branch);
         }

         inline expression_node_ptr operator() (const details::operator_type& operation, expression_node_ptr& b0, expression_node_ptr& b1)
         {
            expression_node_ptr result = error_node();

            if ((0 != b0) && (0 != b1))
            {
               expression_node_ptr branch[2] = { b0, b1 };
               result = expression_generator<Type>::operator()(operation, branch);
               b0 = branch[0];
               b1 = branch[1];
            }

            return result;
         }

         inline expression_node_ptr conditional(expression_node_ptr condition,
                                                expression_node_ptr consequent,
                                                expression_node_ptr alternative) const
         {
            if ((0 == condition) || (0 == consequent))
            {
               details::free_node(*node_allocator_, condition  );
               details::free_node(*node_allocator_, consequent );
               details::free_node(*node_allocator_, alternative);

               return error_node();
            }
            // Can the condition be immediately evaluated? if so optimise.
            else if (details::is_constant_node(condition))
            {
               // True branch
               if (details::is_true(condition))
               {
                  details::free_node(*node_allocator_, condition  );
                  details::free_node(*node_allocator_, alternative);

                  return consequent;
               }
               // False branch
               else
               {
                  details::free_node(*node_allocator_, condition );
                  details::free_node(*node_allocator_, consequent);

                  if (alternative)
                     return alternative;
                  else
                     return node_allocator_->allocate<details::null_node<T> >();
               }
            }
            else if ((0 != consequent) && (0 != alternative))
            {
               return node_allocator_->
                        allocate<conditional_node_t>(condition, consequent, alternative);
            }
            else
               return node_allocator_->
                        allocate<cons_conditional_node_t>(condition, consequent);
         }

         inline expression_node_ptr conditional_vector(expression_node_ptr condition,
                                                       expression_node_ptr consequent,
                                                       expression_node_ptr alternative) const
         {
            if ((0 == condition) || (0 == consequent))
            {
               details::free_node(*node_allocator_, condition  );
               details::free_node(*node_allocator_, consequent );
               details::free_node(*node_allocator_, alternative);

               return error_node();
            }
            // Can the condition be immediately evaluated? if so optimise.
            else if (details::is_constant_node(condition))
            {
               // True branch
               if (details::is_true(condition))
               {
                  details::free_node(*node_allocator_, condition  );
                  details::free_node(*node_allocator_, alternative);

                  return consequent;
               }
               // False branch
               else
               {
                  details::free_node(*node_allocator_, condition );
                  details::free_node(*node_allocator_, consequent);

                  if (alternative)
                     return alternative;
                  else
                     return node_allocator_->allocate<details::null_node<T> >();

               }
            }
            else if ((0 != consequent) && (0 != alternative))
            {
               return node_allocator_->
                        allocate<conditional_vector_node_t>(condition, consequent, alternative);
            }
            else
               return error_node();
         }

         inline loop_runtime_check_ptr get_loop_runtime_check(const loop_runtime_check::loop_types loop_type) const
         {
            if (
                 parser_->loop_runtime_check_ &&
                 (loop_type == (parser_->loop_runtime_check_->loop_set & loop_type))
               )
            {
               return parser_->loop_runtime_check_;
            }

            return loop_runtime_check_ptr(0);
         }

         inline expression_node_ptr while_loop(expression_node_ptr& condition,
                                               expression_node_ptr& branch,
                                               const bool break_continue_present = false) const
         {
            if (!break_continue_present && details::is_constant_node(condition))
            {
               expression_node_ptr result = error_node();
               if (details::is_true(condition))
                  // Infinite loops are not allowed.
                  result = error_node();
               else
                  result = node_allocator_->allocate<details::null_node<Type> >();

               details::free_node(*node_allocator_, condition);
               details::free_node(*node_allocator_, branch   );

               return result;
            }
            else if (details::is_null_node(condition))
            {
               details::free_node(*node_allocator_,condition);

               return branch;
            }

            loop_runtime_check_ptr rtc = get_loop_runtime_check(loop_runtime_check::e_while_loop);

            if (!break_continue_present)
            {
               if (rtc)
                  return node_allocator_->allocate<while_loop_rtc_node_t>
                           (condition, branch,  rtc);
               else
                  return node_allocator_->allocate<while_loop_node_t>
                           (condition, branch);
            }
			return error_node();
         }

         inline expression_node_ptr repeat_until_loop(expression_node_ptr& condition,
                                                      expression_node_ptr& branch,
                                                      const bool break_continue_present = false) const
         {
            if (!break_continue_present && details::is_constant_node(condition))
            {
               if (
                    details::is_true(condition) &&
                    details::is_constant_node(branch)
                  )
               {
                  free_node(*node_allocator_,condition);

                  return branch;
               }

               details::free_node(*node_allocator_, condition);
               details::free_node(*node_allocator_, branch   );

               return error_node();
            }
            else if (details::is_null_node(condition))
            {
               details::free_node(*node_allocator_,condition);

               return branch;
            }

            loop_runtime_check_ptr rtc = get_loop_runtime_check(loop_runtime_check::e_repeat_until_loop);

            if (!break_continue_present)
            {
               if (rtc)
                  return node_allocator_->allocate<repeat_until_loop_rtc_node_t>
                           (condition, branch,  rtc);
               else
                  return node_allocator_->allocate<repeat_until_loop_node_t>
                           (condition, branch);
            }
			return error_node();
         }

         inline expression_node_ptr for_loop(expression_node_ptr& initialiser,
                                             expression_node_ptr& condition,
                                             expression_node_ptr& incrementor,
                                             expression_node_ptr& loop_body,
                                             bool break_continue_present = false) const
         {
            if (!break_continue_present && details::is_constant_node(condition))
            {
               expression_node_ptr result = error_node();

               if (details::is_true(condition))
                  // Infinite loops are not allowed.
                  result = error_node();
               else
                  result = node_allocator_->allocate<details::null_node<Type> >();

               details::free_node(*node_allocator_, initialiser);
               details::free_node(*node_allocator_, condition  );
               details::free_node(*node_allocator_, incrementor);
               details::free_node(*node_allocator_, loop_body  );

               return result;
            }
            else if (details::is_null_node(condition) || (0 == condition))
            {
               details::free_node(*node_allocator_, initialiser);
               details::free_node(*node_allocator_, condition  );
               details::free_node(*node_allocator_, incrementor);

               return loop_body;
            }

            loop_runtime_check_ptr rtc = get_loop_runtime_check(loop_runtime_check::e_for_loop);

            if (!break_continue_present)
            {
               if (rtc)
                  return node_allocator_->allocate<for_loop_rtc_node_t>
                                          (
                                            initialiser,
                                            condition,
                                            incrementor,
                                            loop_body,
                                            rtc
                                          );
               else
                  return node_allocator_->allocate<for_loop_node_t>
                                          (
                                            initialiser,
                                            condition,
                                            incrementor,
                                            loop_body
                                          );
            }
			return error_node();
         }

         template <typename Allocator,
                   template <typename, typename> class Sequence>
         inline expression_node_ptr const_optimise_switch(Sequence<expression_node_ptr,Allocator>& arg_list)
         {
            expression_node_ptr result = error_node();

            for (std::size_t i = 0; i < (arg_list.size() / 2); ++i)
            {
               expression_node_ptr condition  = arg_list[(2 * i)    ];
               expression_node_ptr consequent = arg_list[(2 * i) + 1];

               if ((0 == result) && details::is_true(condition))
               {
                  result = consequent;
                  break;
               }
            }

            if (0 == result)
            {
               result = arg_list.back();
            }

            for (std::size_t i = 0; i < arg_list.size(); ++i)
            {
               expression_node_ptr current_expr = arg_list[i];

               if (current_expr && (current_expr != result))
               {
                  free_node(*node_allocator_,current_expr);
               }
            }

            return result;
         }

         template <typename Allocator,
                   template <typename, typename> class Sequence>
         inline expression_node_ptr const_optimise_mswitch(Sequence<expression_node_ptr,Allocator>& arg_list)
         {
            expression_node_ptr result = error_node();

            for (std::size_t i = 0; i < (arg_list.size() / 2); ++i)
            {
               expression_node_ptr condition  = arg_list[(2 * i)    ];
               expression_node_ptr consequent = arg_list[(2 * i) + 1];

               if (details::is_true(condition))
               {
                  result = consequent;
               }
            }

            if (0 == result)
            {
               T zero = T(0);
               result = node_allocator_->allocate<literal_node_t>(zero);
            }

            for (std::size_t i = 0; i < arg_list.size(); ++i)
            {
               expression_node_ptr& current_expr = arg_list[i];

               if (current_expr && (current_expr != result))
               {
                  details::free_node(*node_allocator_,current_expr);
               }
            }

            return result;
         }

         struct switch_nodes
         {
            typedef std::vector<std::pair<expression_node_ptr,bool> > arg_list_t;

            #define case_stmt(N)                                                         \
            if (is_true(arg[(2 * N)].first)) { return arg[(2 * N) + 1].first->value(); } \

            struct switch_impl_1
            {
               static inline T process(const arg_list_t& arg)
               {
                  case_stmt(0)

                  assert(arg.size() == ((2 * 1) + 1));

                  return arg.back().first->value();
               }
            };

            struct switch_impl_2
            {
               static inline T process(const arg_list_t& arg)
               {
                  case_stmt(0) case_stmt(1)

                  assert(arg.size() == ((2 * 2) + 1));

                  return arg.back().first->value();
               }
            };

            struct switch_impl_3
            {
               static inline T process(const arg_list_t& arg)
               {
                  case_stmt(0) case_stmt(1)
                  case_stmt(2)

                  assert(arg.size() == ((2 * 3) + 1));

                  return arg.back().first->value();
               }
            };

            struct switch_impl_4
            {
               static inline T process(const arg_list_t& arg)
               {
                  case_stmt(0) case_stmt(1)
                  case_stmt(2) case_stmt(3)

                  assert(arg.size() == ((2 * 4) + 1));

                  return arg.back().first->value();
               }
            };

            struct switch_impl_5
            {
               static inline T process(const arg_list_t& arg)
               {
                  case_stmt(0) case_stmt(1)
                  case_stmt(2) case_stmt(3)
                  case_stmt(4)

                  assert(arg.size() == ((2 * 5) + 1));

                  return arg.back().first->value();
               }
            };

            struct switch_impl_6
            {
               static inline T process(const arg_list_t& arg)
               {
                  case_stmt(0) case_stmt(1)
                  case_stmt(2) case_stmt(3)
                  case_stmt(4) case_stmt(5)

                  assert(arg.size() == ((2 * 6) + 1));

                  return arg.back().first->value();
               }
            };

            struct switch_impl_7
            {
               static inline T process(const arg_list_t& arg)
               {
                  case_stmt(0) case_stmt(1)
                  case_stmt(2) case_stmt(3)
                  case_stmt(4) case_stmt(5)
                  case_stmt(6)

                  assert(arg.size() == ((2 * 7) + 1));

                  return arg.back().first->value();
               }
            };

            #undef case_stmt
         };

         template <typename Allocator,
                   template <typename, typename> class Sequence>
         inline expression_node_ptr switch_statement(Sequence<expression_node_ptr,Allocator>& arg_list, const bool default_statement_present)
         {
            if (arg_list.empty())
               return error_node();
            else if (
                      !all_nodes_valid(arg_list) ||
                      (!default_statement_present && (arg_list.size() < 2))
                    )
            {
               details::free_all_nodes(*node_allocator_,arg_list);

               return error_node();
            }
            else if (is_constant_foldable(arg_list))
               return const_optimise_switch(arg_list);

            switch ((arg_list.size() - 1) / 2)
            {
               #define case_stmt(N)                                                       \
               case N :                                                                   \
                  return node_allocator_->                                                \
                            allocate<details::switch_n_node                               \
                              <Type,typename switch_nodes::switch_impl_##N > >(arg_list); \

               case_stmt(1)
               case_stmt(2)
               case_stmt(3)
               case_stmt(4)
               case_stmt(5)
               case_stmt(6)
               case_stmt(7)
               #undef case_stmt

               default : return node_allocator_->allocate<details::switch_node<Type> >(arg_list);
            }
         }

         template <typename Allocator,
                   template <typename, typename> class Sequence>
         inline expression_node_ptr multi_switch_statement(Sequence<expression_node_ptr,Allocator>& arg_list)
         {
            if (!all_nodes_valid(arg_list))
            {
               details::free_all_nodes(*node_allocator_,arg_list);

               return error_node();
            }
            else if (is_constant_foldable(arg_list))
               return const_optimise_mswitch(arg_list);
            else
               return node_allocator_->allocate<details::multi_switch_node<Type> >(arg_list);
         }

         #define unary_opr_switch_statements             \
         case_stmt(details::e_abs   , details::abs_op  ) \
         case_stmt(details::e_acos  , details::acos_op ) \
         case_stmt(details::e_acosh , details::acosh_op) \
         case_stmt(details::e_asin  , details::asin_op ) \
         case_stmt(details::e_asinh , details::asinh_op) \
         case_stmt(details::e_atan  , details::atan_op ) \
         case_stmt(details::e_atanh , details::atanh_op) \
         case_stmt(details::e_ceil  , details::ceil_op ) \
         case_stmt(details::e_cos   , details::cos_op  ) \
         case_stmt(details::e_cosh  , details::cosh_op ) \
         case_stmt(details::e_exp   , details::exp_op  ) \
         case_stmt(details::e_expm1 , details::expm1_op) \
         case_stmt(details::e_floor , details::floor_op) \
         case_stmt(details::e_log   , details::log_op  ) \
         case_stmt(details::e_log10 , details::log10_op) \
         case_stmt(details::e_log2  , details::log2_op ) \
         case_stmt(details::e_log1p , details::log1p_op) \
         case_stmt(details::e_neg   , details::neg_op  ) \
         case_stmt(details::e_pos   , details::pos_op  ) \
         case_stmt(details::e_round , details::round_op) \
         case_stmt(details::e_sin   , details::sin_op  ) \
         case_stmt(details::e_sinc  , details::sinc_op ) \
         case_stmt(details::e_sinh  , details::sinh_op ) \
         case_stmt(details::e_sqrt  , details::sqrt_op ) \
         case_stmt(details::e_tan   , details::tan_op  ) \
         case_stmt(details::e_tanh  , details::tanh_op ) \
         case_stmt(details::e_cot   , details::cot_op  ) \
         case_stmt(details::e_sec   , details::sec_op  ) \
         case_stmt(details::e_csc   , details::csc_op  ) \
         case_stmt(details::e_r2d   , details::r2d_op  ) \
         case_stmt(details::e_d2r   , details::d2r_op  ) \
         case_stmt(details::e_d2g   , details::d2g_op  ) \
         case_stmt(details::e_g2d   , details::g2d_op  ) \
         case_stmt(details::e_notl  , details::notl_op ) \
         case_stmt(details::e_sgn   , details::sgn_op  ) \
         case_stmt(details::e_erf   , details::erf_op  ) \
         case_stmt(details::e_erfc  , details::erfc_op ) \
         case_stmt(details::e_ncdf  , details::ncdf_op ) \
         case_stmt(details::e_frac  , details::frac_op ) \
         case_stmt(details::e_trunc , details::trunc_op) \

         inline expression_node_ptr synthesize_uv_expression(const details::operator_type& operation,
                                                             expression_node_ptr (&branch)[1])
         {
            T& v = static_cast<details::variable_node<T>*>(branch[0])->ref();

            switch (operation)
            {
               #define case_stmt(op0, op1)                                                         \
               case op0 : return node_allocator_->                                                 \
                             allocate<typename details::unary_variable_node<Type,op1<Type> > >(v); \

               unary_opr_switch_statements
               #undef case_stmt
               default : return error_node();
            }
         }

         inline expression_node_ptr synthesize_uvec_expression(const details::operator_type& operation,
                                                               expression_node_ptr (&branch)[1])
         {
            switch (operation)
            {
               #define case_stmt(op0, op1)                                                   \
               case op0 : return node_allocator_->                                           \
                             allocate<typename details::unary_vector_node<Type,op1<Type> > > \
                                (operation, branch[0]);                                      \

               unary_opr_switch_statements
               #undef case_stmt
               default : return error_node();
            }
         }

         inline expression_node_ptr synthesize_unary_expression(const details::operator_type& operation,
                                                                expression_node_ptr (&branch)[1])
         {
            switch (operation)
            {
               #define case_stmt(op0, op1)                                                               \
               case op0 : return node_allocator_->                                                       \
                             allocate<typename details::unary_branch_node<Type,op1<Type> > >(branch[0]); \

               unary_opr_switch_statements
               #undef case_stmt
               default : return error_node();
            }
         }

         inline expression_node_ptr const_optimise_sf3(const details::operator_type& operation,
                                                       expression_node_ptr (&branch)[3])
         {
            expression_node_ptr temp_node = error_node();

            switch (operation)
            {
               #define case_stmt(op)                                                        \
               case details::e_sf##op : temp_node = node_allocator_->                       \
                             allocate<details::sf3_node<Type,details::sf##op##_op<Type> > > \
                                (operation, branch);                                        \
                             break;                                                         \

               case_stmt(00) case_stmt(01) case_stmt(02) case_stmt(03)
               case_stmt(04) case_stmt(05) case_stmt(06) case_stmt(07)
               case_stmt(08) case_stmt(09) case_stmt(10) case_stmt(11)
               case_stmt(12) case_stmt(13) case_stmt(14) case_stmt(15)
               case_stmt(16) case_stmt(17) case_stmt(18) case_stmt(19)
               case_stmt(20) case_stmt(21) case_stmt(22) case_stmt(23)
               case_stmt(24) case_stmt(25) case_stmt(26) case_stmt(27)
               case_stmt(28) case_stmt(29) case_stmt(30) case_stmt(31)
               case_stmt(32) case_stmt(33) case_stmt(34) case_stmt(35)
               case_stmt(36) case_stmt(37) case_stmt(38) case_stmt(39)
               case_stmt(40) case_stmt(41) case_stmt(42) case_stmt(43)
               case_stmt(44) case_stmt(45) case_stmt(46) case_stmt(47)
               #undef case_stmt
               default : return error_node();
            }

            const T v = temp_node->value();

            details::free_node(*node_allocator_,temp_node);

            return node_allocator_->allocate<literal_node_t>(v);
         }

         inline expression_node_ptr varnode_optimise_sf3(const details::operator_type& operation, expression_node_ptr (&branch)[3])
         {
            typedef details::variable_node<Type>* variable_ptr;

            const Type& v0 = static_cast<variable_ptr>(branch[0])->ref();
            const Type& v1 = static_cast<variable_ptr>(branch[1])->ref();
            const Type& v2 = static_cast<variable_ptr>(branch[2])->ref();

            switch (operation)
            {
               #define case_stmt(op)                                                                \
               case details::e_sf##op : return node_allocator_->                                    \
                             allocate_rrr<details::sf3_var_node<Type,details::sf##op##_op<Type> > > \
                                (v0, v1, v2);                                                       \

               case_stmt(00) case_stmt(01) case_stmt(02) case_stmt(03)
               case_stmt(04) case_stmt(05) case_stmt(06) case_stmt(07)
               case_stmt(08) case_stmt(09) case_stmt(10) case_stmt(11)
               case_stmt(12) case_stmt(13) case_stmt(14) case_stmt(15)
               case_stmt(16) case_stmt(17) case_stmt(18) case_stmt(19)
               case_stmt(20) case_stmt(21) case_stmt(22) case_stmt(23)
               case_stmt(24) case_stmt(25) case_stmt(26) case_stmt(27)
               case_stmt(28) case_stmt(29) case_stmt(30) case_stmt(31)
               case_stmt(32) case_stmt(33) case_stmt(34) case_stmt(35)
               case_stmt(36) case_stmt(37) case_stmt(38) case_stmt(39)
               case_stmt(40) case_stmt(41) case_stmt(42) case_stmt(43)
               case_stmt(44) case_stmt(45) case_stmt(46) case_stmt(47)
               #undef case_stmt
               default : return error_node();
            }
         }

         inline expression_node_ptr special_function(const details::operator_type& operation, expression_node_ptr (&branch)[3])
         {
            if (!all_nodes_valid(branch))
               return error_node();
            else if (is_constant_foldable(branch))
               return const_optimise_sf3(operation,branch);
            else if (all_nodes_variables(branch))
               return varnode_optimise_sf3(operation,branch);
            else
            {
               switch (operation)
               {
                  #define case_stmt(op)                                                        \
                  case details::e_sf##op : return node_allocator_->                            \
                                allocate<details::sf3_node<Type,details::sf##op##_op<Type> > > \
                                   (operation, branch);                                        \

                  case_stmt(00) case_stmt(01) case_stmt(02) case_stmt(03)
                  case_stmt(04) case_stmt(05) case_stmt(06) case_stmt(07)
                  case_stmt(08) case_stmt(09) case_stmt(10) case_stmt(11)
                  case_stmt(12) case_stmt(13) case_stmt(14) case_stmt(15)
                  case_stmt(16) case_stmt(17) case_stmt(18) case_stmt(19)
                  case_stmt(20) case_stmt(21) case_stmt(22) case_stmt(23)
                  case_stmt(24) case_stmt(25) case_stmt(26) case_stmt(27)
                  case_stmt(28) case_stmt(29) case_stmt(30) case_stmt(31)
                  case_stmt(32) case_stmt(33) case_stmt(34) case_stmt(35)
                  case_stmt(36) case_stmt(37) case_stmt(38) case_stmt(39)
                  case_stmt(40) case_stmt(41) case_stmt(42) case_stmt(43)
                  case_stmt(44) case_stmt(45) case_stmt(46) case_stmt(47)
                  #undef case_stmt
                  default : return error_node();
               }
            }
         }

         inline expression_node_ptr const_optimise_sf4(const details::operator_type& operation, expression_node_ptr (&branch)[4])
         {
            expression_node_ptr temp_node = error_node();

            switch (operation)
            {
               #define case_stmt(op)                                                                    \
               case details::e_sf##op : temp_node = node_allocator_->                                   \
                                         allocate<details::sf4_node<Type,details::sf##op##_op<Type> > > \
                                            (operation, branch);                                        \
                                        break;                                                          \

               case_stmt(48) case_stmt(49) case_stmt(50) case_stmt(51)
               case_stmt(52) case_stmt(53) case_stmt(54) case_stmt(55)
               case_stmt(56) case_stmt(57) case_stmt(58) case_stmt(59)
               case_stmt(60) case_stmt(61) case_stmt(62) case_stmt(63)
               case_stmt(64) case_stmt(65) case_stmt(66) case_stmt(67)
               case_stmt(68) case_stmt(69) case_stmt(70) case_stmt(71)
               case_stmt(72) case_stmt(73) case_stmt(74) case_stmt(75)
               case_stmt(76) case_stmt(77) case_stmt(78) case_stmt(79)
               case_stmt(80) case_stmt(81) case_stmt(82) case_stmt(83)
               case_stmt(84) case_stmt(85) case_stmt(86) case_stmt(87)
               case_stmt(88) case_stmt(89) case_stmt(90) case_stmt(91)
               case_stmt(92) case_stmt(93) case_stmt(94) case_stmt(95)
               case_stmt(96) case_stmt(97) case_stmt(98) case_stmt(99)
               #undef case_stmt
               default : return error_node();
            }

            const T v = temp_node->value();

            details::free_node(*node_allocator_,temp_node);

            return node_allocator_->allocate<literal_node_t>(v);
         }

         inline expression_node_ptr varnode_optimise_sf4(const details::operator_type& operation, expression_node_ptr (&branch)[4])
         {
            typedef details::variable_node<Type>* variable_ptr;

            const Type& v0 = static_cast<variable_ptr>(branch[0])->ref();
            const Type& v1 = static_cast<variable_ptr>(branch[1])->ref();
            const Type& v2 = static_cast<variable_ptr>(branch[2])->ref();
            const Type& v3 = static_cast<variable_ptr>(branch[3])->ref();

            switch (operation)
            {
               #define case_stmt(op)                                                                 \
               case details::e_sf##op : return node_allocator_->                                     \
                             allocate_rrrr<details::sf4_var_node<Type,details::sf##op##_op<Type> > > \
                                (v0, v1, v2, v3);                                                    \

               case_stmt(48) case_stmt(49) case_stmt(50) case_stmt(51)
               case_stmt(52) case_stmt(53) case_stmt(54) case_stmt(55)
               case_stmt(56) case_stmt(57) case_stmt(58) case_stmt(59)
               case_stmt(60) case_stmt(61) case_stmt(62) case_stmt(63)
               case_stmt(64) case_stmt(65) case_stmt(66) case_stmt(67)
               case_stmt(68) case_stmt(69) case_stmt(70) case_stmt(71)
               case_stmt(72) case_stmt(73) case_stmt(74) case_stmt(75)
               case_stmt(76) case_stmt(77) case_stmt(78) case_stmt(79)
               case_stmt(80) case_stmt(81) case_stmt(82) case_stmt(83)
               case_stmt(84) case_stmt(85) case_stmt(86) case_stmt(87)
               case_stmt(88) case_stmt(89) case_stmt(90) case_stmt(91)
               case_stmt(92) case_stmt(93) case_stmt(94) case_stmt(95)
               case_stmt(96) case_stmt(97) case_stmt(98) case_stmt(99)
               #undef case_stmt
               default : return error_node();
            }
         }

         inline expression_node_ptr special_function(const details::operator_type& operation, expression_node_ptr (&branch)[4])
         {
            if (!all_nodes_valid(branch))
               return error_node();
            else if (is_constant_foldable(branch))
               return const_optimise_sf4(operation,branch);
            else if (all_nodes_variables(branch))
               return varnode_optimise_sf4(operation,branch);
            switch (operation)
            {
               #define case_stmt(op)                                                        \
               case details::e_sf##op : return node_allocator_->                            \
                             allocate<details::sf4_node<Type,details::sf##op##_op<Type> > > \
                                (operation, branch);                                        \

               case_stmt(48) case_stmt(49) case_stmt(50) case_stmt(51)
               case_stmt(52) case_stmt(53) case_stmt(54) case_stmt(55)
               case_stmt(56) case_stmt(57) case_stmt(58) case_stmt(59)
               case_stmt(60) case_stmt(61) case_stmt(62) case_stmt(63)
               case_stmt(64) case_stmt(65) case_stmt(66) case_stmt(67)
               case_stmt(68) case_stmt(69) case_stmt(70) case_stmt(71)
               case_stmt(72) case_stmt(73) case_stmt(74) case_stmt(75)
               case_stmt(76) case_stmt(77) case_stmt(78) case_stmt(79)
               case_stmt(80) case_stmt(81) case_stmt(82) case_stmt(83)
               case_stmt(84) case_stmt(85) case_stmt(86) case_stmt(87)
               case_stmt(88) case_stmt(89) case_stmt(90) case_stmt(91)
               case_stmt(92) case_stmt(93) case_stmt(94) case_stmt(95)
               case_stmt(96) case_stmt(97) case_stmt(98) case_stmt(99)
               #undef case_stmt
               default : return error_node();
            }
         }

         template <typename Allocator,
                   template <typename, typename> class Sequence>
         inline expression_node_ptr const_optimise_varargfunc(const details::operator_type& operation, Sequence<expression_node_ptr,Allocator>& arg_list)
         {
            expression_node_ptr temp_node = error_node();

            switch (operation)
            {
               #define case_stmt(op0, op1)                                                \
               case op0 : temp_node = node_allocator_->                                   \
                                         allocate<details::vararg_node<Type,op1<Type> > > \
                                            (arg_list);                                   \
                          break;                                                          \

               case_stmt(details::e_sum   , details::vararg_add_op  )
               case_stmt(details::e_prod  , details::vararg_mul_op  )
               case_stmt(details::e_avg   , details::vararg_avg_op  )
               case_stmt(details::e_min   , details::vararg_min_op  )
               case_stmt(details::e_max   , details::vararg_max_op  )
               case_stmt(details::e_mand  , details::vararg_mand_op )
               case_stmt(details::e_mor   , details::vararg_mor_op  )
               case_stmt(details::e_multi , details::vararg_multi_op)
               #undef case_stmt
               default : return error_node();
            }

            const T v = temp_node->value();

            details::free_node(*node_allocator_,temp_node);

            return node_allocator_->allocate<literal_node_t>(v);
         }

         inline bool special_one_parameter_vararg(const details::operator_type& operation) const
         {
            return (
                     (details::e_sum  == operation) ||
                     (details::e_prod == operation) ||
                     (details::e_avg  == operation) ||
                     (details::e_min  == operation) ||
                     (details::e_max  == operation)
                   );
         }

         template <typename Allocator,
                   template <typename, typename> class Sequence>
         inline expression_node_ptr varnode_optimise_varargfunc(const details::operator_type& operation, Sequence<expression_node_ptr,Allocator>& arg_list)
         {
            switch (operation)
            {
               #define case_stmt(op0, op1)                                                  \
               case op0 : return node_allocator_->                                          \
                             allocate<details::vararg_varnode<Type,op1<Type> > >(arg_list); \

               case_stmt(details::e_sum   , details::vararg_add_op  )
               case_stmt(details::e_prod  , details::vararg_mul_op  )
               case_stmt(details::e_avg   , details::vararg_avg_op  )
               case_stmt(details::e_min   , details::vararg_min_op  )
               case_stmt(details::e_max   , details::vararg_max_op  )
               case_stmt(details::e_mand  , details::vararg_mand_op )
               case_stmt(details::e_mor   , details::vararg_mor_op  )
               case_stmt(details::e_multi , details::vararg_multi_op)
               #undef case_stmt
               default : return error_node();
            }
         }

         template <typename Allocator,
                   template <typename, typename> class Sequence>
         inline expression_node_ptr vectorize_func(const details::operator_type& operation, Sequence<expression_node_ptr,Allocator>& arg_list)
         {
            if (1 == arg_list.size())
            {
               switch (operation)
               {
                  #define case_stmt(op0, op1)                                                     \
                  case op0 : return node_allocator_->                                             \
                                allocate<details::vectorize_node<Type,op1<Type> > >(arg_list[0]); \

                  case_stmt(details::e_sum  , details::vec_add_op)
                  case_stmt(details::e_prod , details::vec_mul_op)
                  case_stmt(details::e_avg  , details::vec_avg_op)
                  case_stmt(details::e_min  , details::vec_min_op)
                  case_stmt(details::e_max  , details::vec_max_op)
                  #undef case_stmt
                  default : return error_node();
               }
            }
            else
               return error_node();
         }

         template <typename Allocator,
                   template <typename, typename> class Sequence>
         inline expression_node_ptr vararg_function(const details::operator_type& operation, Sequence<expression_node_ptr,Allocator>& arg_list)
         {
            if (!all_nodes_valid(arg_list))
            {
               details::free_all_nodes(*node_allocator_,arg_list);

               return error_node();
            }
            else if (is_constant_foldable(arg_list))
               return const_optimise_varargfunc(operation,arg_list);
            else if ((arg_list.size() == 1) && details::is_ivector_node(arg_list[0]))
               return vectorize_func(operation,arg_list);
            else if ((arg_list.size() == 1) && special_one_parameter_vararg(operation))
               return arg_list[0];
            else if (all_nodes_variables(arg_list))
               return varnode_optimise_varargfunc(operation,arg_list);

            {
               switch (operation)
               {
                  #define case_stmt(op0, op1)                                               \
                  case op0 : return node_allocator_->                                       \
                                allocate<details::vararg_node<Type,op1<Type> > >(arg_list); \

                  case_stmt(details::e_sum   , details::vararg_add_op  )
                  case_stmt(details::e_prod  , details::vararg_mul_op  )
                  case_stmt(details::e_avg   , details::vararg_avg_op  )
                  case_stmt(details::e_min   , details::vararg_min_op  )
                  case_stmt(details::e_max   , details::vararg_max_op  )
                  case_stmt(details::e_mand  , details::vararg_mand_op )
                  case_stmt(details::e_mor   , details::vararg_mor_op  )
                  case_stmt(details::e_multi , details::vararg_multi_op)
                  #undef case_stmt
                  default : return error_node();
               }
            }
         }

         template <std::size_t N>
         inline expression_node_ptr function(ifunction_t* f, expression_node_ptr (&b)[N])
         {
            typedef typename details::function_N_node<T,ifunction_t,N> function_N_node_t;
            expression_node_ptr result = synthesize_expression<function_N_node_t,N>(f,b);

            if (0 == result)
               return error_node();
            else
            {
               // Can the function call be completely optimised?
               if (details::is_constant_node(result))
                  return result;
               else if (!all_nodes_valid(b))
               {
                  details::free_node(*node_allocator_,result);
                  std::fill_n(b, N, reinterpret_cast<expression_node_ptr>(0));

                  return error_node();
               }
               else if (N != f->param_count)
               {
                  details::free_node(*node_allocator_,result);
                  std::fill_n(b, N, reinterpret_cast<expression_node_ptr>(0));

                  return error_node();
               }

               function_N_node_t* func_node_ptr = reinterpret_cast<function_N_node_t*>(result);

               if (!func_node_ptr->init_branches(b))
               {
                  details::free_node(*node_allocator_,result);
                  std::fill_n(b, N, reinterpret_cast<expression_node_ptr>(0));

                  return error_node();
               }

               return result;
            }
         }

         inline expression_node_ptr function(ifunction_t* f)
         {
            typedef typename details::function_N_node<Type,ifunction_t,0> function_N_node_t;
            return node_allocator_->allocate<function_N_node_t>(f);
         }

         inline expression_node_ptr vararg_function_call(ivararg_function_t* vaf,
                                                         std::vector<expression_node_ptr>& arg_list)
         {
            if (!all_nodes_valid(arg_list))
            {
               details::free_all_nodes(*node_allocator_,arg_list);

               return error_node();
            }

            typedef details::vararg_function_node<Type,ivararg_function_t> alloc_type;

            expression_node_ptr result = node_allocator_->allocate<alloc_type>(vaf,arg_list);

            if (
                 !arg_list.empty()        &&
                 !vaf->has_side_effects() &&
                 is_constant_foldable(arg_list)
               )
            {
               const Type v = result->value();
               details::free_node(*node_allocator_,result);
               result = node_allocator_->allocate<literal_node_t>(v);
            }

            parser_->state_.activate_side_effect("vararg_function_call()");

            return result;
         }

         inline expression_node_ptr generic_function_call(igeneric_function_t* gf,
                                                          std::vector<expression_node_ptr>& arg_list,
                                                          const std::size_t& param_seq_index = std::numeric_limits<std::size_t>::max())
         {
            if (!all_nodes_valid(arg_list))
            {
               details::free_all_nodes(*node_allocator_,arg_list);
               return error_node();
            }

            typedef details::generic_function_node     <Type,igeneric_function_t> alloc_type1;
            typedef details::multimode_genfunction_node<Type,igeneric_function_t> alloc_type2;

            const std::size_t no_psi = std::numeric_limits<std::size_t>::max();

            expression_node_ptr result = error_node();

            if (no_psi == param_seq_index)
               result = node_allocator_->allocate<alloc_type1>(arg_list,gf);
            else
               result = node_allocator_->allocate<alloc_type2>(gf, param_seq_index, arg_list);

            alloc_type1* genfunc_node_ptr = static_cast<alloc_type1*>(result);

            if (
                 !arg_list.empty()                  &&
                 !gf->has_side_effects()            &&
                 parser_->state_.type_check_enabled &&
                 is_constant_foldable(arg_list)
               )
            {
               genfunc_node_ptr->init_branches();

               const Type v = result->value();

               details::free_node(*node_allocator_,result);

               return node_allocator_->allocate<literal_node_t>(v);
            }
            else if (genfunc_node_ptr->init_branches())
            {
               parser_->state_.activate_side_effect("generic_function_call()");

               return result;
            }
            else
            {
               details::free_node(*node_allocator_, result);
               details::free_all_nodes(*node_allocator_, arg_list);

               return error_node();
            }
         }

		 inline expression_node_ptr return_call(std::vector<expression_node_ptr>&)
         {
            return error_node();
         }

         inline expression_node_ptr return_envelope(expression_node_ptr,
                                                    results_context_t*,
                                                    bool*&)
         {
            return error_node();
         }

         inline expression_node_ptr vector_element(const std::string& symbol,
                                                   vector_holder_ptr vector_base,
                                                   expression_node_ptr index)
         {
            expression_node_ptr result = error_node();

            if (details::is_constant_node(index))
            {
               std::size_t i = static_cast<std::size_t>(details::numeric::to_int64(index->value()));

               details::free_node(*node_allocator_,index);

               if (vector_base->rebaseable())
               {
                  return node_allocator_->allocate<rebasevector_celem_node_t>(i,vector_base);
               }

               const scope_element& se = parser_->sem_.get_element(symbol,i);

               if (se.index == i)
               {
                  result = se.var_node;
               }
               else
               {
                  scope_element nse;
                  nse.name      = symbol;
                  nse.active    = true;
                  nse.ref_count = 1;
                  nse.type      = scope_element::e_vecelem;
                  nse.index     = i;
                  nse.depth     = parser_->state_.scope_depth;
                  nse.data      = 0;
                  nse.var_node  = node_allocator_->allocate<variable_node_t>((*(*vector_base)[i]));

                  if (!parser_->sem_.add_element(nse))
                  {
                     parser_->set_synthesis_error("Failed to add new local vector element to SEM [1]");

                     parser_->sem_.free_element(nse);

                     result = error_node();
                  }

                  exprtk_debug(("vector_element() - INFO - Added new local vector element: %s\n",nse.name.c_str()));

                  parser_->state_.activate_side_effect("vector_element()");

                  result = nse.var_node;
               }
            }
            else if (vector_base->rebaseable())
               result = node_allocator_->allocate<rebasevector_elem_node_t>(index,vector_base);
            else
               result = node_allocator_->allocate<vector_elem_node_t>(index,vector_base);

            return result;
         }

      private:

         template <std::size_t N, typename NodePtr>
         inline bool is_constant_foldable(NodePtr (&b)[N]) const
         {
            for (std::size_t i = 0; i < N; ++i)
            {
               if (0 == b[i])
                  return false;
               else if (!details::is_constant_node(b[i]))
                  return false;
            }

            return true;
         }

         template <typename NodePtr,
                   typename Allocator,
                   template <typename, typename> class Sequence>
         inline bool is_constant_foldable(const Sequence<NodePtr,Allocator>& b) const
         {
            for (std::size_t i = 0; i < b.size(); ++i)
            {
               if (0 == b[i])
                  return false;
               else if (!details::is_constant_node(b[i]))
                  return false;
            }

            return true;
         }

         void lodge_assignment(symbol_type cst, expression_node_ptr node)
         {
            parser_->state_.activate_side_effect("lodge_assignment()");

            if (!parser_->dec_.collect_assignments())
               return;

            std::string symbol_name;

            switch (cst)
            {
               case e_st_variable : symbol_name = parser_->symtab_store_
                                                     .get_variable_name(node);
                                    break;

               case e_st_vector   : {
                                       typedef details::vector_holder<T> vector_holder_t;

                                       vector_holder_t& vh = static_cast<vector_node_t*>(node)->vec_holder();

                                       symbol_name = parser_->symtab_store_.get_vector_name(&vh);
                                    }
                                    break;

               case e_st_vecelem  : {
                                       typedef details::vector_holder<T> vector_holder_t;

                                       vector_holder_t& vh = static_cast<vector_elem_node_t*>(node)->vec_holder();

                                       symbol_name = parser_->symtab_store_.get_vector_name(&vh);

                                       cst = e_st_vector;
                                    }
                                    break;

               default            : return;
            }

            if (!symbol_name.empty())
            {
               parser_->dec_.add_assignment(symbol_name,cst);
            }
         }

         inline expression_node_ptr synthesize_assignment_expression(const details::operator_type& operation, expression_node_ptr (&branch)[2])
         {
            if (details::is_variable_node(branch[0]))
            {
               lodge_assignment(e_st_variable,branch[0]);

               return synthesize_expression<assignment_node_t,2>(operation,branch);
            }
            else if (details::is_vector_elem_node(branch[0]))
            {
               lodge_assignment(e_st_vecelem,branch[0]);

               return synthesize_expression<assignment_vec_elem_node_t, 2>(operation, branch);
            }
            else if (details::is_rebasevector_elem_node(branch[0]))
            {
               lodge_assignment(e_st_vecelem,branch[0]);

               return synthesize_expression<assignment_rebasevec_elem_node_t, 2>(operation, branch);
            }
            else if (details::is_rebasevector_celem_node(branch[0]))
            {
               lodge_assignment(e_st_vecelem,branch[0]);

               return synthesize_expression<assignment_rebasevec_celem_node_t, 2>(operation, branch);
            }
            else if (details::is_vector_node(branch[0]))
            {
               lodge_assignment(e_st_vector,branch[0]);

               if (details::is_ivector_node(branch[1]))
                  return synthesize_expression<assignment_vecvec_node_t,2>(operation, branch);
               else
                  return synthesize_expression<assignment_vec_node_t,2>(operation, branch);
            }
            else
            {
               parser_->set_synthesis_error("Invalid assignment operation.[1]");

               return error_node();
            }
         }

         inline expression_node_ptr synthesize_assignment_operation_expression(const details::operator_type& operation,
                                                                               expression_node_ptr (&branch)[2])
         {
            if (details::is_variable_node(branch[0]))
            {
               lodge_assignment(e_st_variable,branch[0]);

               switch (operation)
               {
                  #define case_stmt(op0, op1)                                                                 \
                  case op0 : return node_allocator_->                                                         \
                                template allocate_rrr<typename details::assignment_op_node<Type,op1<Type> > > \
                                   (operation, branch[0], branch[1]);                                         \

                  case_stmt(details::e_addass , details::add_op)
                  case_stmt(details::e_subass , details::sub_op)
                  case_stmt(details::e_mulass , details::mul_op)
                  case_stmt(details::e_divass , details::div_op)
                  case_stmt(details::e_modass , details::mod_op)
                  #undef case_stmt
                  default : return error_node();
               }
            }
            else if (details::is_vector_elem_node(branch[0]))
            {
               lodge_assignment(e_st_vecelem,branch[0]);

               switch (operation)
               {
                  #define case_stmt(op0, op1)                                                                           \
                  case op0 : return node_allocator_->                                                                   \
                                 template allocate_rrr<typename details::assignment_vec_elem_op_node<Type,op1<Type> > > \
                                    (operation, branch[0], branch[1]);                                                  \

                  case_stmt(details::e_addass , details::add_op)
                  case_stmt(details::e_subass , details::sub_op)
                  case_stmt(details::e_mulass , details::mul_op)
                  case_stmt(details::e_divass , details::div_op)
                  case_stmt(details::e_modass , details::mod_op)
                  #undef case_stmt
                  default : return error_node();
               }
            }
            else if (details::is_rebasevector_elem_node(branch[0]))
            {
               lodge_assignment(e_st_vecelem,branch[0]);

               switch (operation)
               {
                  #define case_stmt(op0, op1)                                                                                 \
                  case op0 : return node_allocator_->                                                                         \
                                 template allocate_rrr<typename details::assignment_rebasevec_elem_op_node<Type,op1<Type> > > \
                                    (operation, branch[0], branch[1]);                                                        \

                  case_stmt(details::e_addass , details::add_op)
                  case_stmt(details::e_subass , details::sub_op)
                  case_stmt(details::e_mulass , details::mul_op)
                  case_stmt(details::e_divass , details::div_op)
                  case_stmt(details::e_modass , details::mod_op)
                  #undef case_stmt
                  default : return error_node();
               }
            }
            else if (details::is_rebasevector_celem_node(branch[0]))
            {
               lodge_assignment(e_st_vecelem,branch[0]);

               switch (operation)
               {
                  #define case_stmt(op0, op1)                                                                                  \
                  case op0 : return node_allocator_->                                                                          \
                                 template allocate_rrr<typename details::assignment_rebasevec_celem_op_node<Type,op1<Type> > > \
                                    (operation, branch[0], branch[1]);                                                         \

                  case_stmt(details::e_addass , details::add_op)
                  case_stmt(details::e_subass , details::sub_op)
                  case_stmt(details::e_mulass , details::mul_op)
                  case_stmt(details::e_divass , details::div_op)
                  case_stmt(details::e_modass , details::mod_op)
                  #undef case_stmt
                  default : return error_node();
               }
            }
            else if (details::is_vector_node(branch[0]))
            {
               lodge_assignment(e_st_vector,branch[0]);

               if (details::is_ivector_node(branch[1]))
               {
                  switch (operation)
                  {
                     #define case_stmt(op0, op1)                                                                        \
                     case op0 : return node_allocator_->                                                                \
                                   template allocate_rrr<typename details::assignment_vecvec_op_node<Type,op1<Type> > > \
                                      (operation, branch[0], branch[1]);                                                \

                     case_stmt(details::e_addass , details::add_op)
                     case_stmt(details::e_subass , details::sub_op)
                     case_stmt(details::e_mulass , details::mul_op)
                     case_stmt(details::e_divass , details::div_op)
                     case_stmt(details::e_modass , details::mod_op)
                     #undef case_stmt
                     default : return error_node();
                  }
               }
               else
               {
                  switch (operation)
                  {
                     #define case_stmt(op0, op1)                                                                     \
                     case op0 : return node_allocator_->                                                             \
                                   template allocate_rrr<typename details::assignment_vec_op_node<Type,op1<Type> > > \
                                      (operation, branch[0], branch[1]);                                             \

                     case_stmt(details::e_addass , details::add_op)
                     case_stmt(details::e_subass , details::sub_op)
                     case_stmt(details::e_mulass , details::mul_op)
                     case_stmt(details::e_divass , details::div_op)
                     case_stmt(details::e_modass , details::mod_op)
                     #undef case_stmt
                     default : return error_node();
                  }
               }
            }
            else
            {
               parser_->set_synthesis_error("Invalid assignment operation[2]");

               return error_node();
            }
         }

         inline expression_node_ptr synthesize_veceqineqlogic_operation_expression(const details::operator_type& operation,
                                                                                   expression_node_ptr (&branch)[2])
         {
            const bool is_b0_ivec = details::is_ivector_node(branch[0]);
            const bool is_b1_ivec = details::is_ivector_node(branch[1]);

            #define batch_eqineq_logic_case                 \
            case_stmt(details::e_lt    , details::lt_op   ) \
            case_stmt(details::e_lte   , details::lte_op  ) \
            case_stmt(details::e_gt    , details::gt_op   ) \
            case_stmt(details::e_gte   , details::gte_op  ) \
            case_stmt(details::e_eq    , details::eq_op   ) \
            case_stmt(details::e_ne    , details::ne_op   ) \
            case_stmt(details::e_equal , details::equal_op) \
            case_stmt(details::e_and   , details::and_op  ) \
            case_stmt(details::e_nand  , details::nand_op ) \
            case_stmt(details::e_or    , details::or_op   ) \
            case_stmt(details::e_nor   , details::nor_op  ) \
            case_stmt(details::e_xor   , details::xor_op  ) \
            case_stmt(details::e_xnor  , details::xnor_op ) \

            if (is_b0_ivec && is_b1_ivec)
            {
               switch (operation)
               {
                  #define case_stmt(op0, op1)                                                                    \
                  case op0 : return node_allocator_->                                                            \
                                template allocate_rrr<typename details::vec_binop_vecvec_node<Type,op1<Type> > > \
                                   (operation, branch[0], branch[1]);                                            \

                  batch_eqineq_logic_case
                  #undef case_stmt
                  default : return error_node();
               }
            }
            else if (is_b0_ivec && !is_b1_ivec)
            {
               switch (operation)
               {
                  #define case_stmt(op0, op1)                                                                    \
                  case op0 : return node_allocator_->                                                            \
                                template allocate_rrr<typename details::vec_binop_vecval_node<Type,op1<Type> > > \
                                   (operation, branch[0], branch[1]);                                            \

                  batch_eqineq_logic_case
                  #undef case_stmt
                  default : return error_node();
               }
            }
            else if (!is_b0_ivec && is_b1_ivec)
            {
               switch (operation)
               {
                  #define case_stmt(op0, op1)                                                                    \
                  case op0 : return node_allocator_->                                                            \
                                template allocate_rrr<typename details::vec_binop_valvec_node<Type,op1<Type> > > \
                                   (operation, branch[0], branch[1]);                                            \

                  batch_eqineq_logic_case
                  #undef case_stmt
                  default : return error_node();
               }
            }
            else
               return error_node();

            #undef batch_eqineq_logic_case
         }

         inline expression_node_ptr synthesize_vecarithmetic_operation_expression(const details::operator_type& operation,
                                                                                  expression_node_ptr (&branch)[2])
         {
            const bool is_b0_ivec = details::is_ivector_node(branch[0]);
            const bool is_b1_ivec = details::is_ivector_node(branch[1]);

            #define vector_ops                          \
            case_stmt(details::e_add , details::add_op) \
            case_stmt(details::e_sub , details::sub_op) \
            case_stmt(details::e_mul , details::mul_op) \
            case_stmt(details::e_div , details::div_op) \
            case_stmt(details::e_mod , details::mod_op) \

            if (is_b0_ivec && is_b1_ivec)
            {
               switch (operation)
               {
                  #define case_stmt(op0, op1)                                                                    \
                  case op0 : return node_allocator_->                                                            \
                                template allocate_rrr<typename details::vec_binop_vecvec_node<Type,op1<Type> > > \
                                   (operation, branch[0], branch[1]);                                            \

                  vector_ops
                  case_stmt(details::e_pow,details:: pow_op)
                  #undef case_stmt
                  default : return error_node();
               }
            }
            else if (is_b0_ivec && !is_b1_ivec)
            {
               switch (operation)
               {
                  #define case_stmt(op0, op1)                                                                    \
                  case op0 : return node_allocator_->                                                            \
                                template allocate_rrr<typename details::vec_binop_vecval_node<Type,op1<Type> > > \
                                   (operation, branch[0], branch[1]);                                            \

                  vector_ops
                  case_stmt(details::e_pow,details:: pow_op)
                  #undef case_stmt
                  default : return error_node();
               }
            }
            else if (!is_b0_ivec && is_b1_ivec)
            {
               switch (operation)
               {
                  #define case_stmt(op0, op1)                                                                    \
                  case op0 : return node_allocator_->                                                            \
                                template allocate_rrr<typename details::vec_binop_valvec_node<Type,op1<Type> > > \
                                   (operation, branch[0], branch[1]);                                            \

                  vector_ops
                  #undef case_stmt
                  default : return error_node();
               }
            }
            else
               return error_node();

            #undef vector_ops
         }

         inline expression_node_ptr synthesize_swap_expression(expression_node_ptr (&branch)[2])
         {
            const bool v0_is_ivar = details::is_ivariable_node(branch[0]);
            const bool v1_is_ivar = details::is_ivariable_node(branch[1]);

            const bool v0_is_ivec = details::is_ivector_node  (branch[0]);
            const bool v1_is_ivec = details::is_ivector_node  (branch[1]);

            expression_node_ptr result = error_node();

            if (v0_is_ivar && v1_is_ivar)
            {
               typedef details::variable_node<T>* variable_node_ptr;

               variable_node_ptr v0 = variable_node_ptr(0);
               variable_node_ptr v1 = variable_node_ptr(0);

               if (
                    (0 != (v0 = dynamic_cast<variable_node_ptr>(branch[0]))) &&
                    (0 != (v1 = dynamic_cast<variable_node_ptr>(branch[1])))
                  )
               {
                  result = node_allocator_->allocate<details::swap_node<T> >(v0,v1);
               }
               else
                  result = node_allocator_->allocate<details::swap_generic_node<T> >(branch[0],branch[1]);
            }
            else if (v0_is_ivec && v1_is_ivec)
            {
               result = node_allocator_->allocate<details::swap_vecvec_node<T> >(branch[0],branch[1]);
            }
            else
            {
               parser_->set_synthesis_error("Only variables, strings, vectors or vector elements can be swapped");

               return error_node();
            }

            parser_->state_.activate_side_effect("synthesize_swap_expression()");

            return result;
         }

         inline expression_node_ptr synthesize_shortcircuit_expression(const details::operator_type&, expression_node_ptr (&)[2])
         {
            return error_node();
         }

         #define basic_opr_switch_statements         \
         case_stmt(details::e_add , details::add_op) \
         case_stmt(details::e_sub , details::sub_op) \
         case_stmt(details::e_mul , details::mul_op) \
         case_stmt(details::e_div , details::div_op) \
         case_stmt(details::e_mod , details::mod_op) \
         case_stmt(details::e_pow , details::pow_op) \

         #define extended_opr_switch_statements        \
         case_stmt(details::e_lt   , details::lt_op  ) \
         case_stmt(details::e_lte  , details::lte_op ) \
         case_stmt(details::e_gt   , details::gt_op  ) \
         case_stmt(details::e_gte  , details::gte_op ) \
         case_stmt(details::e_eq   , details::eq_op  ) \
         case_stmt(details::e_ne   , details::ne_op  ) \
         case_stmt(details::e_and  , details::and_op ) \
         case_stmt(details::e_nand , details::nand_op) \
         case_stmt(details::e_or   , details::or_op  ) \
         case_stmt(details::e_nor  , details::nor_op ) \
         case_stmt(details::e_xor  , details::xor_op ) \
         case_stmt(details::e_xnor , details::xnor_op) \

         #ifndef exprtk_disable_cardinal_pow_optimisation
         template <typename TType, template <typename, typename> class IPowNode>
         inline expression_node_ptr cardinal_pow_optimisation_impl(const TType& v, const unsigned int& p)
         {
            switch (p)
            {
               #define case_stmt(cp)                                                     \
               case cp : return node_allocator_->                                        \
                            allocate<IPowNode<T,details::numeric::fast_exp<T,cp> > >(v); \

               case_stmt( 1) case_stmt( 2) case_stmt( 3) case_stmt( 4)
               case_stmt( 5) case_stmt( 6) case_stmt( 7) case_stmt( 8)
               case_stmt( 9) case_stmt(10) case_stmt(11) case_stmt(12)
               case_stmt(13) case_stmt(14) case_stmt(15) case_stmt(16)
               case_stmt(17) case_stmt(18) case_stmt(19) case_stmt(20)
               case_stmt(21) case_stmt(22) case_stmt(23) case_stmt(24)
               case_stmt(25) case_stmt(26) case_stmt(27) case_stmt(28)
               case_stmt(29) case_stmt(30) case_stmt(31) case_stmt(32)
               case_stmt(33) case_stmt(34) case_stmt(35) case_stmt(36)
               case_stmt(37) case_stmt(38) case_stmt(39) case_stmt(40)
               case_stmt(41) case_stmt(42) case_stmt(43) case_stmt(44)
               case_stmt(45) case_stmt(46) case_stmt(47) case_stmt(48)
               case_stmt(49) case_stmt(50) case_stmt(51) case_stmt(52)
               case_stmt(53) case_stmt(54) case_stmt(55) case_stmt(56)
               case_stmt(57) case_stmt(58) case_stmt(59) case_stmt(60)
               #undef case_stmt
               default : return error_node();
            }
         }

         inline expression_node_ptr cardinal_pow_optimisation(const T& v, const T& c)
         {
            const bool not_recipricol = (c >= T(0));
            const unsigned int p = static_cast<unsigned int>(details::numeric::to_int32(details::numeric::abs(c)));

            if (0 == p)
               return node_allocator_->allocate_c<literal_node_t>(T(1));
            else if (std::equal_to<T>()(T(2),c))
            {
               return node_allocator_->
                  template allocate_rr<typename details::vov_node<Type,details::mul_op<Type> > >(v,v);
            }
            else
            {
               if (not_recipricol)
                  return cardinal_pow_optimisation_impl<T,details::ipow_node>(v,p);
               else
                  return cardinal_pow_optimisation_impl<T,details::ipowinv_node>(v,p);
            }
         }

         inline bool cardinal_pow_optimisable(const details::operator_type& operation, const T& c) const
         {
            return (details::e_pow == operation) && (details::numeric::abs(c) <= T(60)) && details::numeric::is_integer(c);
         }

         inline expression_node_ptr cardinal_pow_optimisation(expression_node_ptr (&branch)[2])
         {
            const Type c = static_cast<details::literal_node<Type>*>(branch[1])->value();
            const bool not_recipricol = (c >= T(0));
            const unsigned int p = static_cast<unsigned int>(details::numeric::to_int32(details::numeric::abs(c)));

            node_allocator_->free(branch[1]);

            if (0 == p)
            {
               details::free_all_nodes(*node_allocator_, branch);

               return node_allocator_->allocate_c<literal_node_t>(T(1));
            }
            else if (not_recipricol)
               return cardinal_pow_optimisation_impl<expression_node_ptr,details::bipow_node>(branch[0],p);
            else
               return cardinal_pow_optimisation_impl<expression_node_ptr,details::bipowninv_node>(branch[0],p);
         }
         #else
         inline expression_node_ptr cardinal_pow_optimisation(T&, const T&)
         {
            return error_node();
         }

         inline bool cardinal_pow_optimisable(const details::operator_type&, const T&)
         {
            return false;
         }

         inline expression_node_ptr cardinal_pow_optimisation(expression_node_ptr(&)[2])
         {
            return error_node();
         }
         #endif

         struct synthesize_binary_ext_expression
         {
            static inline expression_node_ptr process(expression_generator<Type>& expr_gen,
                                                      const details::operator_type& operation,
                                                      expression_node_ptr (&branch)[2])
            {
               const bool left_neg  = is_neg_unary_node(branch[0]);
               const bool right_neg = is_neg_unary_node(branch[1]);

               if (left_neg && right_neg)
               {
                  if (
                       (details::e_add == operation) ||
                       (details::e_sub == operation) ||
                       (details::e_mul == operation) ||
                       (details::e_div == operation)
                     )
                  {
                     if (
                          !expr_gen.parser_->simplify_unary_negation_branch(branch[0]) ||
                          !expr_gen.parser_->simplify_unary_negation_branch(branch[1])
                        )
                     {
                        details::free_all_nodes(*expr_gen.node_allocator_,branch);

                        return error_node();
                     }
                  }

                  switch (operation)
                  {
                                           // -f(x + 1) + -g(y + 1) --> -(f(x + 1) + g(y + 1))
                     case details::e_add : return expr_gen(details::e_neg,
                                              expr_gen.node_allocator_->
                                                 template allocate<typename details::binary_ext_node<Type,details::add_op<Type> > >
                                                    (branch[0],branch[1]));

                                           // -f(x + 1) - -g(y + 1) --> g(y + 1) - f(x + 1)
                     case details::e_sub : return expr_gen.node_allocator_->
                                              template allocate<typename details::binary_ext_node<Type,details::sub_op<Type> > >
                                                 (branch[1],branch[0]);

                     default             : break;
                  }
               }
               else if (left_neg && !right_neg)
               {
                  if (
                       (details::e_add == operation) ||
                       (details::e_sub == operation) ||
                       (details::e_mul == operation) ||
                       (details::e_div == operation)
                     )
                  {
                     if (!expr_gen.parser_->simplify_unary_negation_branch(branch[0]))
                     {
                        details::free_all_nodes(*expr_gen.node_allocator_,branch);

                        return error_node();
                     }

                     switch (operation)
                     {
                                              // -f(x + 1) + g(y + 1) --> g(y + 1) - f(x + 1)
                        case details::e_add : return expr_gen.node_allocator_->
                                                 template allocate<typename details::binary_ext_node<Type,details::sub_op<Type> > >
                                                   (branch[1], branch[0]);

                                              // -f(x + 1) - g(y + 1) --> -(f(x + 1) + g(y + 1))
                        case details::e_sub : return expr_gen(details::e_neg,
                                                 expr_gen.node_allocator_->
                                                    template allocate<typename details::binary_ext_node<Type,details::add_op<Type> > >
                                                       (branch[0], branch[1]));

                                              // -f(x + 1) * g(y + 1) --> -(f(x + 1) * g(y + 1))
                        case details::e_mul : return expr_gen(details::e_neg,
                                                 expr_gen.node_allocator_->
                                                    template allocate<typename details::binary_ext_node<Type,details::mul_op<Type> > >
                                                       (branch[0], branch[1]));

                                              // -f(x + 1) / g(y + 1) --> -(f(x + 1) / g(y + 1))
                        case details::e_div : return expr_gen(details::e_neg,
                                                 expr_gen.node_allocator_->
                                                    template allocate<typename details::binary_ext_node<Type,details::div_op<Type> > >
                                                       (branch[0], branch[1]));

                        default             : return error_node();
                     }
                  }
               }
               else if (!left_neg && right_neg)
               {
                  if (
                       (details::e_add == operation) ||
                       (details::e_sub == operation) ||
                       (details::e_mul == operation) ||
                       (details::e_div == operation)
                     )
                  {
                     if (!expr_gen.parser_->simplify_unary_negation_branch(branch[1]))
                     {
                        details::free_all_nodes(*expr_gen.node_allocator_,branch);

                        return error_node();
                     }

                     switch (operation)
                     {
                                              // f(x + 1) + -g(y + 1) --> f(x + 1) - g(y + 1)
                        case details::e_add : return expr_gen.node_allocator_->
                                                 template allocate<typename details::binary_ext_node<Type,details::sub_op<Type> > >
                                                   (branch[0], branch[1]);

                                              // f(x + 1) - - g(y + 1) --> f(x + 1) + g(y + 1)
                        case details::e_sub : return expr_gen.node_allocator_->
                                                 template allocate<typename details::binary_ext_node<Type,details::add_op<Type> > >
                                                   (branch[0], branch[1]);

                                              // f(x + 1) * -g(y + 1) --> -(f(x + 1) * g(y + 1))
                        case details::e_mul : return expr_gen(details::e_neg,
                                                 expr_gen.node_allocator_->
                                                    template allocate<typename details::binary_ext_node<Type,details::mul_op<Type> > >
                                                       (branch[0], branch[1]));

                                              // f(x + 1) / -g(y + 1) --> -(f(x + 1) / g(y + 1))
                        case details::e_div : return expr_gen(details::e_neg,
                                                 expr_gen.node_allocator_->
                                                    template allocate<typename details::binary_ext_node<Type,details::div_op<Type> > >
                                                       (branch[0], branch[1]));

                        default             : return error_node();
                     }
                  }
               }

               switch (operation)
               {
                  #define case_stmt(op0, op1)                                                          \
                  case op0 : return expr_gen.node_allocator_->                                         \
                                template allocate<typename details::binary_ext_node<Type,op1<Type> > > \
                                   (branch[0], branch[1]);                                             \

                  basic_opr_switch_statements
                  extended_opr_switch_statements
                  #undef case_stmt
                  default : return error_node();
               }
            }
         };

         struct synthesize_vob_expression
         {
            static inline expression_node_ptr process(expression_generator<Type>& expr_gen,
                                                      const details::operator_type& operation,
                                                      expression_node_ptr (&branch)[2])
            {
               const Type& v = static_cast<details::variable_node<Type>*>(branch[0])->ref();

               if (
                    (details::e_mul == operation) ||
                    (details::e_div == operation)
                  )
               {
                  if (details::is_uv_node(branch[1]))
                  {
                     typedef details::uv_base_node<Type>* uvbn_ptr_t;

                     details::operator_type o = static_cast<uvbn_ptr_t>(branch[1])->operation();

                     if (details::e_neg == o)
                     {
                        const Type& v1 = static_cast<uvbn_ptr_t>(branch[1])->v();

                        details::free_node(*expr_gen.node_allocator_,branch[1]);

                        switch (operation)
                        {
                           case details::e_mul : return expr_gen(details::e_neg,
                                                    expr_gen.node_allocator_->
                                                       template allocate_rr<typename details::
                                                          vov_node<Type,details::mul_op<Type> > >(v,v1));

                           case details::e_div : return expr_gen(details::e_neg,
                                                    expr_gen.node_allocator_->
                                                       template allocate_rr<typename details::
                                                          vov_node<Type,details::div_op<Type> > >(v,v1));

                           default             : break;
                        }
                     }
                  }
               }

               switch (operation)
               {
                  #define case_stmt(op0, op1)                                                      \
                  case op0 : return expr_gen.node_allocator_->                                     \
                                template allocate_rc<typename details::vob_node<Type,op1<Type> > > \
                                   (v, branch[1]);                                                 \

                  basic_opr_switch_statements
                  extended_opr_switch_statements
                  #undef case_stmt
                  default : return error_node();
               }
            }
         };

         struct synthesize_bov_expression
         {
            static inline expression_node_ptr process(expression_generator<Type>& expr_gen,
                                                      const details::operator_type& operation,
                                                      expression_node_ptr (&branch)[2])
            {
               const Type& v = static_cast<details::variable_node<Type>*>(branch[1])->ref();

               if (
                    (details::e_add == operation) ||
                    (details::e_sub == operation) ||
                    (details::e_mul == operation) ||
                    (details::e_div == operation)
                  )
               {
                  if (details::is_uv_node(branch[0]))
                  {
                     typedef details::uv_base_node<Type>* uvbn_ptr_t;

                     details::operator_type o = static_cast<uvbn_ptr_t>(branch[0])->operation();

                     if (details::e_neg == o)
                     {
                        const Type& v0 = static_cast<uvbn_ptr_t>(branch[0])->v();

                        details::free_node(*expr_gen.node_allocator_,branch[0]);

                        switch (operation)
                        {
                           case details::e_add : return expr_gen.node_allocator_->
                                                    template allocate_rr<typename details::
                                                       vov_node<Type,details::sub_op<Type> > >(v,v0);

                           case details::e_sub : return expr_gen(details::e_neg,
                                                    expr_gen.node_allocator_->
                                                       template allocate_rr<typename details::
                                                          vov_node<Type,details::add_op<Type> > >(v0,v));

                           case details::e_mul : return expr_gen(details::e_neg,
                                                    expr_gen.node_allocator_->
                                                       template allocate_rr<typename details::
                                                          vov_node<Type,details::mul_op<Type> > >(v0,v));

                           case details::e_div : return expr_gen(details::e_neg,
                                                    expr_gen.node_allocator_->
                                                       template allocate_rr<typename details::
                                                          vov_node<Type,details::div_op<Type> > >(v0,v));
                           default : break;
                        }
                     }
                  }
               }

               switch (operation)
               {
                  #define case_stmt(op0, op1)                                                      \
                  case op0 : return expr_gen.node_allocator_->                                     \
                                template allocate_cr<typename details::bov_node<Type,op1<Type> > > \
                                   (branch[0], v);                                                 \

                  basic_opr_switch_statements
                  extended_opr_switch_statements
                  #undef case_stmt
                  default : return error_node();
               }
            }
         };

         struct synthesize_cob_expression
         {
            static inline expression_node_ptr process(expression_generator<Type>& expr_gen,
                                                      const details::operator_type& operation,
                                                      expression_node_ptr (&branch)[2])
            {
               const Type c = static_cast<details::literal_node<Type>*>(branch[0])->value();

               details::free_node(*expr_gen.node_allocator_,branch[0]);

               if (std::equal_to<T>()(T(0),c) && (details::e_mul == operation))
               {
                  details::free_node(*expr_gen.node_allocator_,branch[1]);

                  return expr_gen(T(0));
               }
               else if (std::equal_to<T>()(T(0),c) && (details::e_div == operation))
               {
                  details::free_node(*expr_gen.node_allocator_, branch[1]);

                  return expr_gen(T(0));
               }
               else if (std::equal_to<T>()(T(0),c) && (details::e_add == operation))
                  return branch[1];
               else if (std::equal_to<T>()(T(1),c) && (details::e_mul == operation))
                  return branch[1];

               if (details::is_cob_node(branch[1]))
               {
                  // Simplify expressions of the form:
                  // 1. (1 * (2 * (3 * (4 * (5 * (6 * (7 * (8 * (9 + x))))))))) --> 40320 * (9 + x)
                  // 2. (1 + (2 + (3 + (4 + (5 + (6 + (7 + (8 + (9 + x))))))))) --> 45 + x
                  if (
                       (details::e_mul == operation) ||
                       (details::e_add == operation)
                     )
                  {
                     details::cob_base_node<Type>* cobnode = static_cast<details::cob_base_node<Type>*>(branch[1]);

                     if (operation == cobnode->operation())
                     {
                        switch (operation)
                        {
                           case details::e_add : cobnode->set_c(c + cobnode->c()); break;
                           case details::e_mul : cobnode->set_c(c * cobnode->c()); break;
                           default             : return error_node();
                        }

                        return cobnode;
                     }
                  }

                  if (operation == details::e_mul)
                  {
                     details::cob_base_node<Type>* cobnode = static_cast<details::cob_base_node<Type>*>(branch[1]);
                     details::operator_type cob_opr = cobnode->operation();

                     if (
                          (details::e_div == cob_opr) ||
                          (details::e_mul == cob_opr)
                        )
                     {
                        switch (cob_opr)
                        {
                           case details::e_div : cobnode->set_c(c * cobnode->c()); break;
                           case details::e_mul : cobnode->set_c(cobnode->c() / c); break;
                           default             : return error_node();
                        }

                        return cobnode;
                     }
                  }
                  else if (operation == details::e_div)
                  {
                     details::cob_base_node<Type>* cobnode = static_cast<details::cob_base_node<Type>*>(branch[1]);
                     details::operator_type cob_opr = cobnode->operation();

                     if (
                          (details::e_div == cob_opr) ||
                          (details::e_mul == cob_opr)
                        )
                     {
                        details::expression_node<Type>* new_cobnode = error_node();

                        switch (cob_opr)
                        {
                           case details::e_div : new_cobnode = expr_gen.node_allocator_->
                                                    template allocate_tt<typename details::cob_node<Type,details::mul_op<Type> > >
                                                       (c / cobnode->c(), cobnode->move_branch(0));
                                                 break;

                           case details::e_mul : new_cobnode = expr_gen.node_allocator_->
                                                    template allocate_tt<typename details::cob_node<Type,details::div_op<Type> > >
                                                       (c / cobnode->c(), cobnode->move_branch(0));
                                                 break;

                           default             : return error_node();
                        }

                        details::free_node(*expr_gen.node_allocator_,branch[1]);

                        return new_cobnode;
                     }
                  }
               }
               switch (operation)
               {
                  #define case_stmt(op0, op1)                                                      \
                  case op0 : return expr_gen.node_allocator_->                                     \
                                template allocate_tt<typename details::cob_node<Type,op1<Type> > > \
                                   (c,  branch[1]);                                                \

                  basic_opr_switch_statements
                  extended_opr_switch_statements
                  #undef case_stmt
                  default : return error_node();
               }
            }
         };

         struct synthesize_boc_expression
         {
            static inline expression_node_ptr process(expression_generator<Type>& expr_gen,
                                                      const details::operator_type& operation,
                                                      expression_node_ptr (&branch)[2])
            {
               const Type c = static_cast<details::literal_node<Type>*>(branch[1])->value();

               details::free_node(*(expr_gen.node_allocator_), branch[1]);

               if (std::equal_to<T>()(T(0),c) && (details::e_mul == operation))
               {
                  details::free_node(*expr_gen.node_allocator_, branch[0]);

                  return expr_gen(T(0));
               }
               else if (std::equal_to<T>()(T(0),c) && (details::e_div == operation))
               {
                  details::free_node(*expr_gen.node_allocator_, branch[0]);

                  return expr_gen(std::numeric_limits<T>::quiet_NaN());
               }
               else if (std::equal_to<T>()(T(0),c) && (details::e_add == operation))
                  return branch[0];
               else if (std::equal_to<T>()(T(1),c) && (details::e_mul == operation))
                  return branch[0];

               if (details::is_boc_node(branch[0]))
               {
                  // Simplify expressions of the form:
                  // 1. (((((((((x + 9) * 8) * 7) * 6) * 5) * 4) * 3) * 2) * 1) --> (x + 9) * 40320
                  // 2. (((((((((x + 9) + 8) + 7) + 6) + 5) + 4) + 3) + 2) + 1) --> x + 45
                  if (
                       (details::e_mul == operation) ||
                       (details::e_add == operation)
                     )
                  {
                     details::boc_base_node<Type>* bocnode = static_cast<details::boc_base_node<Type>*>(branch[0]);

                     if (operation == bocnode->operation())
                     {
                        switch (operation)
                        {
                           case details::e_add : bocnode->set_c(c + bocnode->c()); break;
                           case details::e_mul : bocnode->set_c(c * bocnode->c()); break;
                           default             : return error_node();
                        }

                        return bocnode;
                     }
                  }
                  else if (operation == details::e_div)
                  {
                     details::boc_base_node<Type>* bocnode = static_cast<details::boc_base_node<Type>*>(branch[0]);
                     details::operator_type        boc_opr = bocnode->operation();

                     if (
                          (details::e_div == boc_opr) ||
                          (details::e_mul == boc_opr)
                        )
                     {
                        switch (boc_opr)
                        {
                           case details::e_div : bocnode->set_c(c * bocnode->c()); break;
                           case details::e_mul : bocnode->set_c(bocnode->c() / c); break;
                           default             : return error_node();
                        }

                        return bocnode;
                     }
                  }
                  else if (operation == details::e_pow)
                  {
                     // (v ^ c0) ^ c1 --> v ^(c0 * c1)
                     details::boc_base_node<Type>* bocnode = static_cast<details::boc_base_node<Type>*>(branch[0]);
                     details::operator_type        boc_opr = bocnode->operation();

                     if (details::e_pow == boc_opr)
                     {
                        bocnode->set_c(bocnode->c() * c);

                        return bocnode;
                     }
                  }
               }

               switch (operation)
               {
                  #define case_stmt(op0, op1)                                                      \
                  case op0 : return expr_gen.node_allocator_->                                     \
                                template allocate_cr<typename details::boc_node<Type,op1<Type> > > \
                                   (branch[0], c);                                                 \

                  basic_opr_switch_statements
                  extended_opr_switch_statements
                  #undef case_stmt
                  default : return error_node();
               }
            }
         };

         struct synthesize_cocob_expression
         {
            static inline expression_node_ptr process(expression_generator<Type>& expr_gen,
                                                      const details::operator_type& operation,
                                                      expression_node_ptr (&branch)[2])
            {
               expression_node_ptr result = error_node();

               // (cob) o c --> cob
               if (details::is_cob_node(branch[0]))
               {
                  details::cob_base_node<Type>* cobnode = static_cast<details::cob_base_node<Type>*>(branch[0]);

                  const Type c = static_cast<details::literal_node<Type>*>(branch[1])->value();

                  if (std::equal_to<T>()(T(0),c) && (details::e_mul == operation))
                  {
                     details::free_node(*expr_gen.node_allocator_, branch[0]);
                     details::free_node(*expr_gen.node_allocator_, branch[1]);

                     return expr_gen(T(0));
                  }
                  else if (std::equal_to<T>()(T(0),c) && (details::e_div == operation))
                  {
                     details::free_node(*expr_gen.node_allocator_, branch[0]);
                     details::free_node(*expr_gen.node_allocator_, branch[1]);

                     return expr_gen(T(std::numeric_limits<T>::quiet_NaN()));
                  }
                  else if (std::equal_to<T>()(T(0),c) && (details::e_add == operation))
                  {
                     details::free_node(*expr_gen.node_allocator_, branch[1]);

                     return branch[0];
                  }
                  else if (std::equal_to<T>()(T(1),c) && (details::e_mul == operation))
                  {
                     details::free_node(*expr_gen.node_allocator_, branch[1]);

                     return branch[0];
                  }
                  else if (std::equal_to<T>()(T(1),c) && (details::e_div == operation))
                  {
                     details::free_node(*expr_gen.node_allocator_, branch[1]);

                     return branch[0];
                  }

                  const bool op_addsub = (details::e_add == cobnode->operation()) ||
                                         (details::e_sub == cobnode->operation()) ;

                  if (op_addsub)
                  {
                     switch (operation)
                     {
                        case details::e_add : cobnode->set_c(cobnode->c() + c); break;
                        case details::e_sub : cobnode->set_c(cobnode->c() - c); break;
                        default             : return error_node();
                     }

                     result = cobnode;
                  }
                  else if (details::e_mul == cobnode->operation())
                  {
                     switch (operation)
                     {
                        case details::e_mul : cobnode->set_c(cobnode->c() * c); break;
                        case details::e_div : cobnode->set_c(cobnode->c() / c); break;
                        default             : return error_node();
                     }

                     result = cobnode;
                  }
                  else if (details::e_div == cobnode->operation())
                  {
                     if (details::e_mul == operation)
                     {
                        cobnode->set_c(cobnode->c() * c);
                        result = cobnode;
                     }
                     else if (details::e_div == operation)
                     {
                        result = expr_gen.node_allocator_->
                                    template allocate_tt<typename details::cob_node<Type,details::div_op<Type> > >
                                       (cobnode->c() / c, cobnode->move_branch(0));

                        details::free_node(*expr_gen.node_allocator_, branch[0]);
                     }
                  }

                  if (result)
                  {
                     details::free_node(*expr_gen.node_allocator_,branch[1]);
                  }
               }

               // c o (cob) --> cob
               else if (details::is_cob_node(branch[1]))
               {
                  details::cob_base_node<Type>* cobnode = static_cast<details::cob_base_node<Type>*>(branch[1]);

                  const Type c = static_cast<details::literal_node<Type>*>(branch[0])->value();

                  if (std::equal_to<T>()(T(0),c) && (details::e_mul == operation))
                  {
                     details::free_node(*expr_gen.node_allocator_, branch[0]);
                     details::free_node(*expr_gen.node_allocator_, branch[1]);

                     return expr_gen(T(0));
                  }
                  else if (std::equal_to<T>()(T(0),c) && (details::e_div == operation))
                  {
                     details::free_node(*expr_gen.node_allocator_, branch[0]);
                     details::free_node(*expr_gen.node_allocator_, branch[1]);

                     return expr_gen(T(0));
                  }
                  else if (std::equal_to<T>()(T(0),c) && (details::e_add == operation))
                  {
                     details::free_node(*expr_gen.node_allocator_, branch[0]);

                     return branch[1];
                  }
                  else if (std::equal_to<T>()(T(1),c) && (details::e_mul == operation))
                  {
                     details::free_node(*expr_gen.node_allocator_, branch[0]);

                     return branch[1];
                  }

                  if (details::e_add == cobnode->operation())
                  {
                     if (details::e_add == operation)
                     {
                        cobnode->set_c(c + cobnode->c());
                        result = cobnode;
                     }
                     else if (details::e_sub == operation)
                     {
                        result = expr_gen.node_allocator_->
                                    template allocate_tt<typename details::cob_node<Type,details::sub_op<Type> > >
                                       (c - cobnode->c(), cobnode->move_branch(0));

                        details::free_node(*expr_gen.node_allocator_,branch[1]);
                     }
                  }
                  else if (details::e_sub == cobnode->operation())
                  {
                     if (details::e_add == operation)
                     {
                        cobnode->set_c(c + cobnode->c());
                        result = cobnode;
                     }
                     else if (details::e_sub == operation)
                     {
                        result = expr_gen.node_allocator_->
                                    template allocate_tt<typename details::cob_node<Type,details::add_op<Type> > >
                                       (c - cobnode->c(), cobnode->move_branch(0));

                        details::free_node(*expr_gen.node_allocator_,branch[1]);
                     }
                  }
                  else if (details::e_mul == cobnode->operation())
                  {
                     if (details::e_mul == operation)
                     {
                        cobnode->set_c(c * cobnode->c());
                        result = cobnode;
                     }
                     else if (details::e_div == operation)
                     {
                        result = expr_gen.node_allocator_->
                                    template allocate_tt<typename details::cob_node<Type,details::div_op<Type> > >
                                       (c / cobnode->c(), cobnode->move_branch(0));

                        details::free_node(*expr_gen.node_allocator_,branch[1]);
                     }
                  }
                  else if (details::e_div == cobnode->operation())
                  {
                     if (details::e_mul == operation)
                     {
                        cobnode->set_c(c * cobnode->c());
                        result = cobnode;
                     }
                     else if (details::e_div == operation)
                     {
                        result = expr_gen.node_allocator_->
                                    template allocate_tt<typename details::cob_node<Type,details::mul_op<Type> > >
                                       (c / cobnode->c(), cobnode->move_branch(0));

                        details::free_node(*expr_gen.node_allocator_,branch[1]);
                     }
                  }

                  if (result)
                  {
                     details::free_node(*expr_gen.node_allocator_,branch[0]);
                  }
               }

               return result;
            }
         };

         struct synthesize_coboc_expression
         {
            static inline expression_node_ptr process(expression_generator<Type>& expr_gen,
                                                      const details::operator_type& operation,
                                                      expression_node_ptr (&branch)[2])
            {
               expression_node_ptr result = error_node();

               // (boc) o c --> boc
               if (details::is_boc_node(branch[0]))
               {
                  details::boc_base_node<Type>* bocnode = static_cast<details::boc_base_node<Type>*>(branch[0]);

                  const Type c = static_cast<details::literal_node<Type>*>(branch[1])->value();

                  if (details::e_add == bocnode->operation())
                  {
                     switch (operation)
                     {
                        case details::e_add : bocnode->set_c(bocnode->c() + c); break;
                        case details::e_sub : bocnode->set_c(bocnode->c() - c); break;
                        default             : return error_node();
                     }

                     result = bocnode;
                  }
                  else if (details::e_mul == bocnode->operation())
                  {
                     switch (operation)
                     {
                        case details::e_mul : bocnode->set_c(bocnode->c() * c); break;
                        case details::e_div : bocnode->set_c(bocnode->c() / c); break;
                        default             : return error_node();
                     }

                     result = bocnode;
                  }
                  else if (details::e_sub == bocnode->operation())
                  {
                     if (details::e_add == operation)
                     {
                        result = expr_gen.node_allocator_->
                                    template allocate_tt<typename details::boc_node<Type,details::add_op<Type> > >
                                       (bocnode->move_branch(0), c - bocnode->c());

                        details::free_node(*expr_gen.node_allocator_,branch[0]);
                     }
                     else if (details::e_sub == operation)
                     {
                        bocnode->set_c(bocnode->c() + c);
                        result = bocnode;
                     }
                  }
                  else if (details::e_div == bocnode->operation())
                  {
                     switch (operation)
                     {
                        case details::e_div : bocnode->set_c(bocnode->c() * c); break;
                        case details::e_mul : bocnode->set_c(bocnode->c() / c); break;
                        default             : return error_node();
                     }

                     result = bocnode;
                  }

                  if (result)
                  {
                     details::free_node(*expr_gen.node_allocator_, branch[1]);
                  }
               }

               // c o (boc) --> boc
               else if (details::is_boc_node(branch[1]))
               {
                  details::boc_base_node<Type>* bocnode = static_cast<details::boc_base_node<Type>*>(branch[1]);

                  const Type c = static_cast<details::literal_node<Type>*>(branch[0])->value();

                  if (details::e_add == bocnode->operation())
                  {
                     if (details::e_add == operation)
                     {
                        bocnode->set_c(c + bocnode->c());
                        result = bocnode;
                     }
                     else if (details::e_sub == operation)
                     {
                        result = expr_gen.node_allocator_->
                                    template allocate_tt<typename details::cob_node<Type,details::sub_op<Type> > >
                                       (c - bocnode->c(), bocnode->move_branch(0));

                        details::free_node(*expr_gen.node_allocator_,branch[1]);
                     }
                  }
                  else if (details::e_sub == bocnode->operation())
                  {
                     if (details::e_add == operation)
                     {
                        result = expr_gen.node_allocator_->
                                    template allocate_tt<typename details::boc_node<Type,details::add_op<Type> > >
                                       (bocnode->move_branch(0), c - bocnode->c());

                        details::free_node(*expr_gen.node_allocator_,branch[1]);
                     }
                     else if (details::e_sub == operation)
                     {
                        result = expr_gen.node_allocator_->
                                    template allocate_tt<typename details::cob_node<Type,details::sub_op<Type> > >
                                       (c + bocnode->c(), bocnode->move_branch(0));

                        details::free_node(*expr_gen.node_allocator_,branch[1]);
                     }
                  }
                  else if (details::e_mul == bocnode->operation())
                  {
                     if (details::e_mul == operation)
                     {
                        bocnode->set_c(c * bocnode->c());
                        result = bocnode;
                     }
                     else if (details::e_div == operation)
                     {
                        result = expr_gen.node_allocator_->
                                    template allocate_tt<typename details::cob_node<Type,details::div_op<Type> > >
                                       (c / bocnode->c(), bocnode->move_branch(0));

                        details::free_node(*expr_gen.node_allocator_,branch[1]);
                     }
                  }
                  else if (details::e_div == bocnode->operation())
                  {
                     if (details::e_mul == operation)
                     {
                        bocnode->set_c(bocnode->c() / c);
                        result = bocnode;
                     }
                     else if (details::e_div == operation)
                     {
                        result = expr_gen.node_allocator_->
                                    template allocate_tt<typename details::cob_node<Type,details::div_op<Type> > >
                                       (c * bocnode->c(), bocnode->move_branch(0));

                        details::free_node(*expr_gen.node_allocator_,branch[1]);
                     }
                  }

                  if (result)
                  {
                     details::free_node(*expr_gen.node_allocator_,branch[0]);
                  }
               }

               return result;
            }
         };

         inline expression_node_ptr synthesize_uvouv_expression(const details::operator_type& operation, expression_node_ptr (&branch)[2])
         {
            // Definition: uv o uv
            details::operator_type o0 = static_cast<details::uv_base_node<Type>*>(branch[0])->operation();
            details::operator_type o1 = static_cast<details::uv_base_node<Type>*>(branch[1])->operation();
            const Type& v0 = static_cast<details::uv_base_node<Type>*>(branch[0])->v();
            const Type& v1 = static_cast<details::uv_base_node<Type>*>(branch[1])->v();
            unary_functor_t u0 = reinterpret_cast<unary_functor_t> (0);
            unary_functor_t u1 = reinterpret_cast<unary_functor_t> (0);
            binary_functor_t f = reinterpret_cast<binary_functor_t>(0);

            if (!valid_operator(o0,u0))
               return error_node();
            else if (!valid_operator(o1,u1))
               return error_node();
            else if (!valid_operator(operation,f))
               return error_node();

            expression_node_ptr result = error_node();

            if (
                 (details::e_neg == o0) &&
                 (details::e_neg == o1)
               )
            {
               switch (operation)
               {
                  // (-v0 + -v1) --> -(v0 + v1)
                  case details::e_add : result = (*this)(details::e_neg,
                                                    node_allocator_->
                                                       allocate_rr<typename details::
                                                          vov_node<Type,details::add_op<Type> > >(v0, v1));
                                        exprtk_debug(("(-v0 + -v1) --> -(v0 + v1)\n"));
                                        break;

                  // (-v0 - -v1) --> (v1 - v0)
                  case details::e_sub : result = node_allocator_->
                                                    allocate_rr<typename details::
                                                       vov_node<Type,details::sub_op<Type> > >(v1, v0);
                                        exprtk_debug(("(-v0 - -v1) --> (v1 - v0)\n"));
                                        break;

                  // (-v0 * -v1) --> (v0 * v1)
                  case details::e_mul : result = node_allocator_->
                                                    allocate_rr<typename details::
                                                       vov_node<Type,details::mul_op<Type> > >(v0, v1);
                                        exprtk_debug(("(-v0 * -v1) --> (v0 * v1)\n"));
                                        break;

                  // (-v0 / -v1) --> (v0 / v1)
                  case details::e_div : result = node_allocator_->
                                                    allocate_rr<typename details::
                                                       vov_node<Type,details::div_op<Type> > >(v0, v1);
                                        exprtk_debug(("(-v0 / -v1) --> (v0 / v1)\n"));
                                        break;

                  default             : break;
               }
            }

            if (0 == result)
            {
               result = node_allocator_->
                            allocate_rrrrr<typename details::uvouv_node<Type> >(v0, v1, u0, u1, f);
            }

            details::free_all_nodes(*node_allocator_,branch);
            return result;
         }

         #undef basic_opr_switch_statements
         #undef extended_opr_switch_statements
         #undef unary_opr_switch_statements
		  
         inline expression_node_ptr synthesize_string_expression(const details::operator_type&, expression_node_ptr (&branch)[2])
         {
            details::free_all_nodes(*node_allocator_,branch);
            return error_node();
         }

         inline expression_node_ptr synthesize_string_expression(const details::operator_type&, expression_node_ptr (&branch)[3])
         {
            details::free_all_nodes(*node_allocator_,branch);
            return error_node();
         }

         inline expression_node_ptr synthesize_null_expression(const details::operator_type& operation, expression_node_ptr (&branch)[2])
         {
            /*
             Note: The following are the type promotion rules
             that relate to operations that include 'null':
             0. null ==/!=     null --> true false
             1. null operation null --> null
             2. x    ==/!=     null --> true/false
             3. null ==/!=     x    --> true/false
             4. x   operation  null --> x
             5. null operation x    --> x
            */

            typedef typename details::null_eq_node<T> nulleq_node_t;

            const bool b0_null = details::is_null_node(branch[0]);
            const bool b1_null = details::is_null_node(branch[1]);

            if (b0_null && b1_null)
            {
               expression_node_ptr result = error_node();

               if (details::e_eq == operation)
                  result = node_allocator_->allocate_c<literal_node_t>(T(1));
               else if (details::e_ne == operation)
                  result = node_allocator_->allocate_c<literal_node_t>(T(0));

               if (result)
               {
                  details::free_node(*node_allocator_,branch[0]);
                  details::free_node(*node_allocator_,branch[1]);

                  return result;
               }

               details::free_node(*node_allocator_,branch[1]);

               return branch[0];
            }
            else if (details::e_eq == operation)
            {
               expression_node_ptr result = node_allocator_->
                                                allocate_rc<nulleq_node_t>(branch[b0_null ? 0 : 1],true);

               details::free_node(*node_allocator_,branch[b0_null ? 1 : 0]);

               return result;
            }
            else if (details::e_ne == operation)
            {
               expression_node_ptr result = node_allocator_->
                                                allocate_rc<nulleq_node_t>(branch[b0_null ? 0 : 1],false);

               details::free_node(*node_allocator_,branch[b0_null ? 1 : 0]);

               return result;
            }
            else if (b0_null)
            {
               details::free_node(*node_allocator_,branch[0]);
               branch[0] = branch[1];
               branch[1] = error_node();
            }
            else if (b1_null)
            {
               details::free_node(*node_allocator_,branch[1]);
               branch[1] = error_node();
            }

            if (
                 (details::e_add == operation) || (details::e_sub == operation) ||
                 (details::e_mul == operation) || (details::e_div == operation) ||
                 (details::e_mod == operation) || (details::e_pow == operation)
               )
            {
               return branch[0];
            }

            details::free_node(*node_allocator_, branch[0]);

            if (
                 (details::e_lt    == operation) || (details::e_lte  == operation) ||
                 (details::e_gt    == operation) || (details::e_gte  == operation) ||
                 (details::e_and   == operation) || (details::e_nand == operation) ||
                 (details::e_or    == operation) || (details::e_nor  == operation) ||
                 (details::e_xor   == operation) || (details::e_xnor == operation) ||
                 (details::e_in    == operation) || (details::e_like == operation) ||
                 (details::e_ilike == operation)
               )
            {
               return node_allocator_->allocate_c<literal_node_t>(T(0));
            }

            return node_allocator_->allocate<details::null_node<Type> >();
         }

         template <typename NodeType, std::size_t N>
         inline expression_node_ptr synthesize_expression(const details::operator_type& operation, expression_node_ptr (&branch)[N])
         {
            if (
                 (details::e_in    == operation) ||
                 (details::e_like  == operation) ||
                 (details::e_ilike == operation)
               )
            {
               free_all_nodes(*node_allocator_,branch);

               return error_node();
            }
            else if (!details::all_nodes_valid<N>(branch))
            {
               free_all_nodes(*node_allocator_,branch);

               return error_node();
            }
            else if ((details::e_default != operation))
            {
               // Attempt simple constant folding optimisation.
               expression_node_ptr expression_point = node_allocator_->allocate<NodeType>(operation,branch);

               if (is_constant_foldable<N>(branch))
               {
                  const Type v = expression_point->value();
                  details::free_node(*node_allocator_,expression_point);

                  return node_allocator_->allocate<literal_node_t>(v);
               }
               else
                  return expression_point;
            }
            else
               return error_node();
         }

         template <typename NodeType, std::size_t N>
         inline expression_node_ptr synthesize_expression(F* f, expression_node_ptr (&branch)[N])
         {
            if (!details::all_nodes_valid<N>(branch))
            {
               free_all_nodes(*node_allocator_,branch);

               return error_node();
            }

            typedef typename details::function_N_node<T,ifunction_t,N> function_N_node_t;

            // Attempt simple constant folding optimisation.

            expression_node_ptr expression_point = node_allocator_->allocate<NodeType>(f);
            function_N_node_t* func_node_ptr = dynamic_cast<function_N_node_t*>(expression_point);

            if (0 == func_node_ptr)
            {
               free_all_nodes(*node_allocator_,branch);

               return error_node();
            }
            else
               func_node_ptr->init_branches(branch);

            if (is_constant_foldable<N>(branch) && !f->has_side_effects())
            {
               Type v = expression_point->value();
               details::free_node(*node_allocator_,expression_point);

               return node_allocator_->allocate<literal_node_t>(v);
            }

            parser_->state_.activate_side_effect("synthesize_expression(function<NT,N>)");

            return expression_point;
         }

         bool                     strength_reduction_enabled_;
         details::node_allocator* node_allocator_;
         synthesize_map_t         synthesize_map_;
         unary_op_map_t*          unary_op_map_;
         binary_op_map_t*         binary_op_map_;
         inv_binary_op_map_t*     inv_binary_op_map_;
         sf3_map_t*               sf3_map_;
         sf4_map_t*               sf4_map_;
         parser_t*                parser_;
      }; // class expression_generator

      inline void set_error(const parser_error::type& error_type)
      {
         error_list_.push_back(error_type);
      }

      inline void remove_last_error()
      {
         if (!error_list_.empty())
         {
            error_list_.pop_back();
         }
      }

      inline void set_synthesis_error(const std::string& synthesis_error_message)
      {
         if (synthesis_error_.empty())
         {
            synthesis_error_ = synthesis_error_message;
         }
      }

      inline void register_local_vars(expression<T>& e)
      {
         for (std::size_t i = 0; i < sem_.size(); ++i)
         {
            scope_element& se = sem_.get_element(i);

            if (
                 (scope_element::e_variable == se.type) ||
                 (scope_element::e_vecelem  == se.type)
               )
            {
               if (se.var_node)
               {
                  e.register_local_var(se.var_node);
               }

               if (se.data)
               {
                  e.register_local_data(se.data, 1, 0);
               }
            }
            else if (scope_element::e_vector == se.type)
            {
               if (se.vec_node)
               {
                  e.register_local_var(se.vec_node);
               }

               if (se.data)
               {
                  e.register_local_data(se.data, se.size, 1);
               }
            }

            se.var_node  = 0;
            se.vec_node  = 0;
            se.data      = 0;
            se.ref_count = 0;
            se.active    = false;
         }
      }

      inline void register_return_results(expression<T>& e)
      {
         e.register_return_results(results_context_);
         results_context_ = 0;
      }

      inline void load_unary_operations_map(unary_op_map_t& m)
      {
         #define register_unary_op(Op, UnaryFunctor)            \
         m.insert(std::make_pair(Op,UnaryFunctor<T>::process)); \

         register_unary_op(details::e_abs   , details::abs_op  )
         register_unary_op(details::e_acos  , details::acos_op )
         register_unary_op(details::e_acosh , details::acosh_op)
         register_unary_op(details::e_asin  , details::asin_op )
         register_unary_op(details::e_asinh , details::asinh_op)
         register_unary_op(details::e_atanh , details::atanh_op)
         register_unary_op(details::e_ceil  , details::ceil_op )
         register_unary_op(details::e_cos   , details::cos_op  )
         register_unary_op(details::e_cosh  , details::cosh_op )
         register_unary_op(details::e_exp   , details::exp_op  )
         register_unary_op(details::e_expm1 , details::expm1_op)
         register_unary_op(details::e_floor , details::floor_op)
         register_unary_op(details::e_log   , details::log_op  )
         register_unary_op(details::e_log10 , details::log10_op)
         register_unary_op(details::e_log2  , details::log2_op )
         register_unary_op(details::e_log1p , details::log1p_op)
         register_unary_op(details::e_neg   , details::neg_op  )
         register_unary_op(details::e_pos   , details::pos_op  )
         register_unary_op(details::e_round , details::round_op)
         register_unary_op(details::e_sin   , details::sin_op  )
         register_unary_op(details::e_sinc  , details::sinc_op )
         register_unary_op(details::e_sinh  , details::sinh_op )
         register_unary_op(details::e_sqrt  , details::sqrt_op )
         register_unary_op(details::e_tan   , details::tan_op  )
         register_unary_op(details::e_tanh  , details::tanh_op )
         register_unary_op(details::e_cot   , details::cot_op  )
         register_unary_op(details::e_sec   , details::sec_op  )
         register_unary_op(details::e_csc   , details::csc_op  )
         register_unary_op(details::e_r2d   , details::r2d_op  )
         register_unary_op(details::e_d2r   , details::d2r_op  )
         register_unary_op(details::e_d2g   , details::d2g_op  )
         register_unary_op(details::e_g2d   , details::g2d_op  )
         register_unary_op(details::e_notl  , details::notl_op )
         register_unary_op(details::e_sgn   , details::sgn_op  )
         register_unary_op(details::e_erf   , details::erf_op  )
         register_unary_op(details::e_erfc  , details::erfc_op )
         register_unary_op(details::e_ncdf  , details::ncdf_op )
         register_unary_op(details::e_frac  , details::frac_op )
         register_unary_op(details::e_trunc , details::trunc_op)
         #undef register_unary_op
      }

      inline void load_binary_operations_map(binary_op_map_t& m)
      {
         typedef typename binary_op_map_t::value_type value_type;

         #define register_binary_op(Op, BinaryFunctor)       \
         m.insert(value_type(Op,BinaryFunctor<T>::process)); \

         register_binary_op(details::e_add  , details::add_op )
         register_binary_op(details::e_sub  , details::sub_op )
         register_binary_op(details::e_mul  , details::mul_op )
         register_binary_op(details::e_div  , details::div_op )
         register_binary_op(details::e_mod  , details::mod_op )
         register_binary_op(details::e_pow  , details::pow_op )
         register_binary_op(details::e_lt   , details::lt_op  )
         register_binary_op(details::e_lte  , details::lte_op )
         register_binary_op(details::e_gt   , details::gt_op  )
         register_binary_op(details::e_gte  , details::gte_op )
         register_binary_op(details::e_eq   , details::eq_op  )
         register_binary_op(details::e_ne   , details::ne_op  )
         register_binary_op(details::e_and  , details::and_op )
         register_binary_op(details::e_nand , details::nand_op)
         register_binary_op(details::e_or   , details::or_op  )
         register_binary_op(details::e_nor  , details::nor_op )
         register_binary_op(details::e_xor  , details::xor_op )
         register_binary_op(details::e_xnor , details::xnor_op)
         #undef register_binary_op
      }

      inline void load_inv_binary_operations_map(inv_binary_op_map_t& m)
      {
         typedef typename inv_binary_op_map_t::value_type value_type;

         #define register_binary_op(Op, BinaryFunctor)       \
         m.insert(value_type(BinaryFunctor<T>::process,Op)); \

         register_binary_op(details::e_add  , details::add_op )
         register_binary_op(details::e_sub  , details::sub_op )
         register_binary_op(details::e_mul  , details::mul_op )
         register_binary_op(details::e_div  , details::div_op )
         register_binary_op(details::e_mod  , details::mod_op )
         register_binary_op(details::e_pow  , details::pow_op )
         register_binary_op(details::e_lt   , details::lt_op  )
         register_binary_op(details::e_lte  , details::lte_op )
         register_binary_op(details::e_gt   , details::gt_op  )
         register_binary_op(details::e_gte  , details::gte_op )
         register_binary_op(details::e_eq   , details::eq_op  )
         register_binary_op(details::e_ne   , details::ne_op  )
         register_binary_op(details::e_and  , details::and_op )
         register_binary_op(details::e_nand , details::nand_op)
         register_binary_op(details::e_or   , details::or_op  )
         register_binary_op(details::e_nor  , details::nor_op )
         register_binary_op(details::e_xor  , details::xor_op )
         register_binary_op(details::e_xnor , details::xnor_op)
         #undef register_binary_op
      }

      inline void load_sf3_map(sf3_map_t& sf3_map)
      {
         typedef std::pair<trinary_functor_t,details::operator_type> pair_t;

         #define register_sf3(Op)                                                                             \
         sf3_map[details::sf##Op##_op<T>::id()] = pair_t(details::sf##Op##_op<T>::process,details::e_sf##Op); \

         register_sf3(00) register_sf3(01) register_sf3(02) register_sf3(03)
         register_sf3(04) register_sf3(05) register_sf3(06) register_sf3(07)
         register_sf3(08) register_sf3(09) register_sf3(10) register_sf3(11)
         register_sf3(12) register_sf3(13) register_sf3(14) register_sf3(15)
         register_sf3(16) register_sf3(17) register_sf3(18) register_sf3(19)
         register_sf3(20) register_sf3(21) register_sf3(22) register_sf3(23)
         register_sf3(24) register_sf3(25) register_sf3(26) register_sf3(27)
         register_sf3(28) register_sf3(29) register_sf3(30)
         #undef register_sf3

         #define register_sf3_extid(Id, Op)                                        \
         sf3_map[Id] = pair_t(details::sf##Op##_op<T>::process,details::e_sf##Op); \

         register_sf3_extid("(t-t)-t",23)  // (t-t)-t --> t-(t+t)
         #undef register_sf3_extid
      }

      inline void load_sf4_map(sf4_map_t& sf4_map)
      {
         typedef std::pair<quaternary_functor_t,details::operator_type> pair_t;

         #define register_sf4(Op)                                                                             \
         sf4_map[details::sf##Op##_op<T>::id()] = pair_t(details::sf##Op##_op<T>::process,details::e_sf##Op); \

         register_sf4(48) register_sf4(49) register_sf4(50) register_sf4(51)
         register_sf4(52) register_sf4(53) register_sf4(54) register_sf4(55)
         register_sf4(56) register_sf4(57) register_sf4(58) register_sf4(59)
         register_sf4(60) register_sf4(61) register_sf4(62) register_sf4(63)
         register_sf4(64) register_sf4(65) register_sf4(66) register_sf4(67)
         register_sf4(68) register_sf4(69) register_sf4(70) register_sf4(71)
         register_sf4(72) register_sf4(73) register_sf4(74) register_sf4(75)
         register_sf4(76) register_sf4(77) register_sf4(78) register_sf4(79)
         register_sf4(80) register_sf4(81) register_sf4(82) register_sf4(83)
         #undef register_sf4

         #define register_sf4ext(Op)                                                                                    \
         sf4_map[details::sfext##Op##_op<T>::id()] = pair_t(details::sfext##Op##_op<T>::process,details::e_sf4ext##Op); \

         register_sf4ext(00) register_sf4ext(01) register_sf4ext(02) register_sf4ext(03)
         register_sf4ext(04) register_sf4ext(05) register_sf4ext(06) register_sf4ext(07)
         register_sf4ext(08) register_sf4ext(09) register_sf4ext(10) register_sf4ext(11)
         register_sf4ext(12) register_sf4ext(13) register_sf4ext(14) register_sf4ext(15)
         register_sf4ext(16) register_sf4ext(17) register_sf4ext(18) register_sf4ext(19)
         register_sf4ext(20) register_sf4ext(21) register_sf4ext(22) register_sf4ext(23)
         register_sf4ext(24) register_sf4ext(25) register_sf4ext(26) register_sf4ext(27)
         register_sf4ext(28) register_sf4ext(29) register_sf4ext(30) register_sf4ext(31)
         register_sf4ext(32) register_sf4ext(33) register_sf4ext(34) register_sf4ext(35)
         register_sf4ext(36) register_sf4ext(36) register_sf4ext(38) register_sf4ext(39)
         register_sf4ext(40) register_sf4ext(41) register_sf4ext(42) register_sf4ext(43)
         register_sf4ext(44) register_sf4ext(45) register_sf4ext(46) register_sf4ext(47)
         register_sf4ext(48) register_sf4ext(49) register_sf4ext(50) register_sf4ext(51)
         register_sf4ext(52) register_sf4ext(53) register_sf4ext(54) register_sf4ext(55)
         register_sf4ext(56) register_sf4ext(57) register_sf4ext(58) register_sf4ext(59)
         register_sf4ext(60) register_sf4ext(61)
         #undef register_sf4ext
      }

      inline results_context_t& results_ctx()
      {
         if (0 == results_context_)
         {
            results_context_ = new results_context_t();
         }

         return (*results_context_);
      }

      inline void return_cleanup()
      {
      }

   private:

      parser(const parser<T>&) exprtk_delete;
      parser<T>& operator=(const parser<T>&) exprtk_delete;

      settings_store settings_;
      expression_generator<T> expression_generator_;
      details::node_allocator node_allocator_;
      symtab_store symtab_store_;
      dependent_entity_collector dec_;
      std::deque<parser_error::type> error_list_;
      std::deque<bool> brkcnt_list_;
      parser_state state_;
      bool resolve_unknown_symbol_;
      results_context_t* results_context_;
      unknown_symbol_resolver* unknown_symbol_resolver_;
      unknown_symbol_resolver default_usr_;
      base_ops_map_t base_ops_map_;
      unary_op_map_t unary_op_map_;
      binary_op_map_t binary_op_map_;
      inv_binary_op_map_t inv_binary_op_map_;
      sf3_map_t sf3_map_;
      sf4_map_t sf4_map_;
      std::string synthesis_error_;
      scope_element_manager sem_;

      lexer::helper::helper_assembly helper_assembly_;

      lexer::helper::commutative_inserter       commutative_inserter_;
      lexer::helper::operator_joiner            operator_joiner_2_;
      lexer::helper::operator_joiner            operator_joiner_3_;
      lexer::helper::symbol_replacer            symbol_replacer_;
      lexer::helper::bracket_checker            bracket_checker_;
      lexer::helper::numeric_checker            numeric_checker_;
      lexer::helper::sequence_validator         sequence_validator_;
      lexer::helper::sequence_validator_3tokens sequence_validator_3tkns_;

      loop_runtime_check_ptr loop_runtime_check_;

      template <typename ParserType>
      friend void details::disable_type_checking(ParserType& p);
   }; // class parser

   namespace details
   {
      template <typename T>
      struct collector_helper
      {
         typedef exprtk::symbol_table<T> symbol_table_t;
         typedef exprtk::expression<T>   expression_t;
         typedef exprtk::parser<T>       parser_t;
         typedef typename parser_t::dependent_entity_collector::symbol_t symbol_t;
         typedef typename parser_t::unknown_symbol_resolver usr_t;

         struct resolve_as_vector : public parser_t::unknown_symbol_resolver
         {
            typedef exprtk::parser<T> parser_t;

            resolve_as_vector()
            : usr_t(usr_t::e_usrmode_extended)
            {}

            virtual bool process(const std::string& unknown_symbol,
                                 symbol_table_t& symbol_table,
                                 std::string&)
            {
               static T v[1];
               symbol_table.add_vector(unknown_symbol,v);
               return true;
            }
         };

         static inline bool collection_pass(const std::string& expression_string,
                                            std::set<std::string>& symbol_set,
                                            const bool collect_variables,
                                            const bool collect_functions,
                                            const bool vector_pass,
                                            symbol_table_t& ext_symbol_table)
         {
            symbol_table_t symbol_table;
            expression_t   expression;
            parser_t       parser;

            resolve_as_vector vect_resolver;

            expression.register_symbol_table(symbol_table    );
            expression.register_symbol_table(ext_symbol_table);

            if (vector_pass)
               parser.enable_unknown_symbol_resolver(&vect_resolver);
            else
               parser.enable_unknown_symbol_resolver();

            if (collect_variables)
               parser.dec().collect_variables() = true;

            if (collect_functions)
               parser.dec().collect_functions() = true;

            bool pass_result = false;

            details::disable_type_checking(parser);

            if (parser.compile(expression_string, expression))
            {
               pass_result = true;

               std::deque<symbol_t> symb_list;
               parser.dec().symbols(symb_list);

               for (std::size_t i = 0; i < symb_list.size(); ++i)
               {
                  symbol_set.insert(symb_list[i].first);
               }
            }

            return pass_result;
         }
      };
   }

   template <typename Allocator,
             template <typename, typename> class Sequence>
   inline bool collect_variables(const std::string& expression,
                                 Sequence<std::string, Allocator>& symbol_list)
   {
      typedef double T;
      typedef details::collector_helper<T> collect_t;

      collect_t::symbol_table_t null_symbol_table;

      std::set<std::string> symbol_set;

      const bool variable_pass = collect_t::collection_pass
                                    (expression, symbol_set, true, false, false, null_symbol_table);
      const bool vector_pass   = collect_t::collection_pass
                                    (expression, symbol_set, true, false,  true, null_symbol_table);

      if (!variable_pass && !vector_pass)
         return false;

      std::set<std::string>::iterator itr = symbol_set.begin();

      while (symbol_set.end() != itr)
      {
         symbol_list.push_back(*itr);
         ++itr;
      }

      return true;
   }

   template <typename T,
             typename Allocator,
             template <typename, typename> class Sequence>
   inline bool collect_variables(const std::string& expression,
                                 exprtk::symbol_table<T>& extrnl_symbol_table,
                                 Sequence<std::string, Allocator>& symbol_list)
   {
      typedef details::collector_helper<T> collect_t;

      std::set<std::string> symbol_set;

      const bool variable_pass = collect_t::collection_pass
                                    (expression, symbol_set, true, false, false, extrnl_symbol_table);
      const bool vector_pass   = collect_t::collection_pass
                                    (expression, symbol_set, true, false,  true, extrnl_symbol_table);

      if (!variable_pass && !vector_pass)
         return false;

      std::set<std::string>::iterator itr = symbol_set.begin();

      while (symbol_set.end() != itr)
      {
         symbol_list.push_back(*itr);
         ++itr;
      }

      return true;
   }

   template <typename Allocator,
             template <typename, typename> class Sequence>
   inline bool collect_functions(const std::string& expression,
                                 Sequence<std::string, Allocator>& symbol_list)
   {
      typedef double T;
      typedef details::collector_helper<T> collect_t;

      collect_t::symbol_table_t null_symbol_table;

      std::set<std::string> symbol_set;

      const bool variable_pass = collect_t::collection_pass
                                    (expression, symbol_set, false, true, false, null_symbol_table);
      const bool vector_pass   = collect_t::collection_pass
                                    (expression, symbol_set, false, true,  true, null_symbol_table);

      if (!variable_pass && !vector_pass)
         return false;

      std::set<std::string>::iterator itr = symbol_set.begin();

      while (symbol_set.end() != itr)
      {
         symbol_list.push_back(*itr);
         ++itr;
      }

      return true;
   }

   template <typename T,
             typename Allocator,
             template <typename, typename> class Sequence>
   inline bool collect_functions(const std::string& expression,
                                 exprtk::symbol_table<T>& extrnl_symbol_table,
                                 Sequence<std::string, Allocator>& symbol_list)
   {
      typedef details::collector_helper<T> collect_t;

      std::set<std::string> symbol_set;

      const bool variable_pass = collect_t::collection_pass
                                    (expression, symbol_set, false, true, false, extrnl_symbol_table);
      const bool vector_pass   = collect_t::collection_pass
                                    (expression, symbol_set, false, true,  true, extrnl_symbol_table);

      if (!variable_pass && !vector_pass)
         return false;

      std::set<std::string>::iterator itr = symbol_set.begin();

      while (symbol_set.end() != itr)
      {
         symbol_list.push_back(*itr);
         ++itr;
      }

      return true;
   }

   template <typename T>
   inline T integrate(const expression<T>& e,
                      T& x,
                      const T& r0, const T& r1,
                      const std::size_t number_of_intervals = 1000000)
   {
      if (r0 > r1)
         return T(0);

      const T h = (r1 - r0) / (T(2) * number_of_intervals);
      T total_area = T(0);

      for (std::size_t i = 0; i < number_of_intervals; ++i)
      {
         x = r0 + T(2) * i * h;
         const T y0 = e.value(); x += h;
         const T y1 = e.value(); x += h;
         const T y2 = e.value(); x += h;
         total_area += h * (y0 + T(4) * y1 + y2) / T(3);
      }

      return total_area;
   }

   template <typename T>
   inline T integrate(const expression<T>& e,
                      const std::string& variable_name,
                      const T& r0, const T& r1,
                      const std::size_t number_of_intervals = 1000000)
   {
      const symbol_table<T>& sym_table = e.get_symbol_table();

      if (!sym_table.valid())
         return std::numeric_limits<T>::quiet_NaN();

      details::variable_node<T>* var = sym_table.get_variable(variable_name);

      if (var)
      {
         T& x = var->ref();
         const T x_original = x;
         const T result = integrate(e, x, r0, r1, number_of_intervals);
         x = x_original;

         return result;
      }
      else
         return std::numeric_limits<T>::quiet_NaN();
   }

   template <typename T>
   inline T derivative(const expression<T>& e,
                       T& x,
                       const T& h = T(0.00000001))
   {
      const T x_init = x;
      const T _2h    = T(2) * h;

      x = x_init + _2h;
      const T y0 = e.value();
      x = x_init + h;
      const T y1 = e.value();
      x = x_init - h;
      const T y2 = e.value();
      x = x_init - _2h;
      const T y3 = e.value();
      x = x_init;

      return (-y0 + T(8) * (y1 - y2) + y3) / (T(12) * h);
   }

   template <typename T>
   inline T second_derivative(const expression<T>& e,
                              T& x,
                              const T& h = T(0.00001))
   {
      const T x_init = x;
      const T _2h    = T(2) * h;

      const T y = e.value();
      x = x_init + _2h;
      const T y0 = e.value();
      x = x_init + h;
      const T y1 = e.value();
      x = x_init - h;
      const T y2 = e.value();
      x = x_init - _2h;
      const T y3 = e.value();
      x = x_init;

      return (-y0 + T(16) * (y1 + y2) - T(30) * y - y3) / (T(12) * h * h);
   }

   template <typename T>
   inline T third_derivative(const expression<T>& e,
                             T& x,
                             const T& h = T(0.0001))
   {
      const T x_init = x;
      const T _2h    = T(2) * h;

      x = x_init + _2h;
      const T y0 = e.value();
      x = x_init + h;
      const T y1 = e.value();
      x = x_init - h;
      const T y2 = e.value();
      x = x_init - _2h;
      const T y3 = e.value();
      x = x_init;

      return (y0 + T(2) * (y2 - y1) - y3) / (T(2) * h * h * h);
   }

   template <typename T>
   inline T derivative(const expression<T>& e,
                       const std::string& variable_name,
                       const T& h = T(0.00000001))
   {
      const symbol_table<T>& sym_table = e.get_symbol_table();

      if (!sym_table.valid())
      {
         return std::numeric_limits<T>::quiet_NaN();
      }

      details::variable_node<T>* var = sym_table.get_variable(variable_name);

      if (var)
      {
         T& x = var->ref();
         const T x_original = x;
         const T result = derivative(e, x, h);
         x = x_original;

         return result;
      }
      else
         return std::numeric_limits<T>::quiet_NaN();
   }

   template <typename T>
   inline T second_derivative(const expression<T>& e,
                              const std::string& variable_name,
                              const T& h = T(0.00001))
   {
      const symbol_table<T>& sym_table = e.get_symbol_table();

      if (!sym_table.valid())
      {
         return std::numeric_limits<T>::quiet_NaN();
      }

      details::variable_node<T>* var = sym_table.get_variable(variable_name);

      if (var)
      {
         T& x = var->ref();
         const T x_original = x;
         const T result = second_derivative(e, x, h);
         x = x_original;

         return result;
      }
      else
         return std::numeric_limits<T>::quiet_NaN();
   }

   template <typename T>
   inline T third_derivative(const expression<T>& e,
                             const std::string& variable_name,
                             const T& h = T(0.0001))
   {
      const symbol_table<T>& sym_table = e.get_symbol_table();

      if (!sym_table.valid())
      {
         return std::numeric_limits<T>::quiet_NaN();
      }

      details::variable_node<T>* var = sym_table.get_variable(variable_name);

      if (var)
      {
         T& x = var->ref();
         const T x_original = x;
         const T result = third_derivative(e, x, h);
         x = x_original;

         return result;
      }
      else
         return std::numeric_limits<T>::quiet_NaN();
   }

   /*
      Note: The following 'compute' routines are simple helpers,
      for quickly setting up the required pieces of code in order
      to evaluate an expression. By virtue of how they operate
      there will be an overhead with regards to their setup and
      teardown and hence should not be used in time critical
      sections of code.
      Furthermore they only assume a small sub set of variables,
      no string variables or user defined functions.
   */
   template <typename T>
   inline bool compute(const std::string& expression_string, T& result)
   {
      // No variables
      symbol_table<T> symbol_table;
      symbol_table.add_constants();

      expression<T> expression;
      expression.register_symbol_table(symbol_table);

      parser<T> parser;

      if (parser.compile(expression_string,expression))
      {
         result = expression.value();

         return true;
      }
      else
         return false;
   }

   template <typename T>
   inline bool compute(const std::string& expression_string,
                       const T& x,
                       T& result)
   {
      // Only 'x'
      static const std::string x_var("x");

      symbol_table<T> symbol_table;
      symbol_table.add_constants();
      symbol_table.add_constant(x_var,x);

      expression<T> expression;
      expression.register_symbol_table(symbol_table);

      parser<T> parser;

      if (parser.compile(expression_string,expression))
      {
         result = expression.value();

         return true;
      }
      else
         return false;
   }

   template <typename T>
   inline bool compute(const std::string& expression_string,
                       const T&x, const T& y,
                       T& result)
   {
      // Only 'x' and 'y'
      static const std::string x_var("x");
      static const std::string y_var("y");

      symbol_table<T> symbol_table;
      symbol_table.add_constants();
      symbol_table.add_constant(x_var,x);
      symbol_table.add_constant(y_var,y);

      expression<T> expression;
      expression.register_symbol_table(symbol_table);

      parser<T> parser;

      if (parser.compile(expression_string,expression))
      {
         result = expression.value();

         return true;
      }
      else
         return false;
   }

   template <typename T>
   inline bool compute(const std::string& expression_string,
                       const T& x, const T& y, const T& z,
                       T& result)
   {
      // Only 'x', 'y' or 'z'
      static const std::string x_var("x");
      static const std::string y_var("y");
      static const std::string z_var("z");

      symbol_table<T> symbol_table;
      symbol_table.add_constants();
      symbol_table.add_constant(x_var,x);
      symbol_table.add_constant(y_var,y);
      symbol_table.add_constant(z_var,z);

      expression<T> expression;
      expression.register_symbol_table(symbol_table);

      parser<T> parser;

      if (parser.compile(expression_string,expression))
      {
         result = expression.value();

         return true;
      }
      else
         return false;
   }

   template <typename T, std::size_t N>
   class polynomial : public ifunction<T>
   {
   private:

      template <typename Type, std::size_t NumberOfCoefficients>
      struct poly_impl { };

      template <typename Type>
      struct poly_impl <Type,12>
      {
         static inline T evaluate(const Type x,
                                  const Type c12, const Type c11, const Type c10, const Type c9, const Type c8,
                                  const Type  c7, const Type  c6, const Type  c5, const Type c4, const Type c3,
                                  const Type  c2, const Type  c1, const Type  c0)
         {
            // p(x) = c_12x^12 + c_11x^11 + c_10x^10 + c_9x^9 + c_8x^8 + c_7x^7 + c_6x^6 + c_5x^5 + c_4x^4 + c_3x^3 + c_2x^2 + c_1x^1 + c_0x^0
            return ((((((((((((c12 * x + c11) * x + c10) * x + c9) * x + c8) * x + c7) * x + c6) * x + c5) * x + c4) * x + c3) * x + c2) * x + c1) * x + c0);
         }
      };

      template <typename Type>
      struct poly_impl <Type,11>
      {
         static inline T evaluate(const Type x,
                                  const Type c11, const Type c10, const Type c9, const Type c8, const Type c7,
                                  const Type c6,  const Type  c5, const Type c4, const Type c3, const Type c2,
                                  const Type c1,  const Type  c0)
         {
            // p(x) = c_11x^11 + c_10x^10 + c_9x^9 + c_8x^8 + c_7x^7 + c_6x^6 + c_5x^5 + c_4x^4 + c_3x^3 + c_2x^2 + c_1x^1 + c_0x^0
            return (((((((((((c11 * x + c10) * x + c9) * x + c8) * x + c7) * x + c6) * x + c5) * x + c4) * x + c3) * x + c2) * x + c1) * x + c0);
         }
      };

      template <typename Type>
      struct poly_impl <Type,10>
      {
         static inline T evaluate(const Type x,
                                  const Type c10, const Type c9, const Type c8, const Type c7, const Type c6,
                                  const Type c5,  const Type c4, const Type c3, const Type c2, const Type c1,
                                  const Type c0)
         {
            // p(x) = c_10x^10 + c_9x^9 + c_8x^8 + c_7x^7 + c_6x^6 + c_5x^5 + c_4x^4 + c_3x^3 + c_2x^2 + c_1x^1 + c_0x^0
            return ((((((((((c10 * x + c9) * x + c8) * x + c7) * x + c6) * x + c5) * x + c4) * x + c3) * x + c2) * x + c1) * x + c0);
         }
      };

      template <typename Type>
      struct poly_impl <Type,9>
      {
         static inline T evaluate(const Type x,
                                  const Type c9, const Type c8, const Type c7, const Type c6, const Type c5,
                                  const Type c4, const Type c3, const Type c2, const Type c1, const Type c0)
         {
            // p(x) = c_9x^9 + c_8x^8 + c_7x^7 + c_6x^6 + c_5x^5 + c_4x^4 + c_3x^3 + c_2x^2 + c_1x^1 + c_0x^0
            return (((((((((c9 * x + c8) * x + c7) * x + c6) * x + c5) * x + c4) * x + c3) * x + c2) * x + c1) * x + c0);
         }
      };

      template <typename Type>
      struct poly_impl <Type,8>
      {
         static inline T evaluate(const Type x,
                                  const Type c8, const Type c7, const Type c6, const Type c5, const Type c4,
                                  const Type c3, const Type c2, const Type c1, const Type c0)
         {
            // p(x) = c_8x^8 + c_7x^7 + c_6x^6 + c_5x^5 + c_4x^4 + c_3x^3 + c_2x^2 + c_1x^1 + c_0x^0
            return ((((((((c8 * x + c7) * x + c6) * x + c5) * x + c4) * x + c3) * x + c2) * x + c1) * x + c0);
         }
      };

      template <typename Type>
      struct poly_impl <Type,7>
      {
         static inline T evaluate(const Type x,
                                  const Type c7, const Type c6, const Type c5, const Type c4, const Type c3,
                                  const Type c2, const Type c1, const Type c0)
         {
            // p(x) = c_7x^7 + c_6x^6 + c_5x^5 + c_4x^4 + c_3x^3 + c_2x^2 + c_1x^1 + c_0x^0
            return (((((((c7 * x + c6) * x + c5) * x + c4) * x + c3) * x + c2) * x + c1) * x + c0);
         }
      };

      template <typename Type>
      struct poly_impl <Type,6>
      {
         static inline T evaluate(const Type x,
                                  const Type c6, const Type c5, const Type c4, const Type c3, const Type c2,
                                  const Type c1, const Type c0)
         {
            // p(x) = c_6x^6 + c_5x^5 + c_4x^4 + c_3x^3 + c_2x^2 + c_1x^1 + c_0x^0
            return ((((((c6 * x + c5) * x + c4) * x + c3) * x + c2) * x + c1) * x + c0);
         }
      };

      template <typename Type>
      struct poly_impl <Type,5>
      {
         static inline T evaluate(const Type x,
                                  const Type c5, const Type c4, const Type c3, const Type c2,
                                  const Type c1, const Type c0)
         {
            // p(x) = c_5x^5 + c_4x^4 + c_3x^3 + c_2x^2 + c_1x^1 + c_0x^0
            return (((((c5 * x + c4) * x + c3) * x + c2) * x + c1) * x + c0);
         }
      };

      template <typename Type>
      struct poly_impl <Type,4>
      {
         static inline T evaluate(const Type x, const Type c4, const Type c3, const Type c2, const Type c1, const Type c0)
         {
            // p(x) = c_4x^4 + c_3x^3 + c_2x^2 + c_1x^1 + c_0x^0
            return ((((c4 * x + c3) * x + c2) * x + c1) * x + c0);
         }
      };

      template <typename Type>
      struct poly_impl <Type,3>
      {
         static inline T evaluate(const Type x, const Type c3, const Type c2, const Type c1, const Type c0)
         {
            // p(x) = c_3x^3 + c_2x^2 + c_1x^1 + c_0x^0
            return (((c3 * x + c2) * x + c1) * x + c0);
         }
      };

      template <typename Type>
      struct poly_impl <Type,2>
      {
         static inline T evaluate(const Type x, const Type c2, const Type c1, const Type c0)
         {
            // p(x) = c_2x^2 + c_1x^1 + c_0x^0
            return ((c2 * x + c1) * x + c0);
         }
      };

      template <typename Type>
      struct poly_impl <Type,1>
      {
         static inline T evaluate(const Type x, const Type c1, const Type c0)
         {
            // p(x) = c_1x^1 + c_0x^0
            return (c1 * x + c0);
         }
      };

   public:

      using ifunction<T>::operator();

      polynomial()
      : ifunction<T>((N+2 <= 20) ? (N + 2) : std::numeric_limits<std::size_t>::max())
      {
         disable_has_side_effects(*this);
      }

      virtual ~polynomial() {}

      #define poly_rtrn(NN) \
      return (NN != N) ? std::numeric_limits<T>::quiet_NaN() :

      inline virtual T operator() (const T& x, const T& c1, const T& c0)
      {
         poly_rtrn(1) (poly_impl<T,1>::evaluate(x, c1, c0));
      }

      inline virtual T operator() (const T& x, const T& c2, const T& c1, const T& c0)
      {
         poly_rtrn(2) (poly_impl<T,2>::evaluate(x, c2, c1, c0));
      }

      inline virtual T operator() (const T& x, const T& c3, const T& c2, const T& c1, const T& c0)
      {
         poly_rtrn(3) (poly_impl<T,3>::evaluate(x, c3, c2, c1, c0));
      }

      inline virtual T operator() (const T& x, const T& c4, const T& c3, const T& c2, const T& c1,
                                   const T& c0)
      {
         poly_rtrn(4) (poly_impl<T,4>::evaluate(x, c4, c3, c2, c1, c0));
      }

      inline virtual T operator() (const T& x, const T& c5, const T& c4, const T& c3, const T& c2,
                                   const T& c1, const T& c0)
      {
         poly_rtrn(5) (poly_impl<T,5>::evaluate(x, c5, c4, c3, c2, c1, c0));
      }

      inline virtual T operator() (const T& x, const T& c6, const T& c5, const T& c4, const T& c3,
                                   const T& c2, const T& c1, const T& c0)
      {
         poly_rtrn(6) (poly_impl<T,6>::evaluate(x, c6, c5, c4, c3, c2, c1, c0));
      }

      inline virtual T operator() (const T& x, const T& c7, const T& c6, const T& c5, const T& c4,
                                   const T& c3, const T& c2, const T& c1, const T& c0)
      {
         poly_rtrn(7) (poly_impl<T,7>::evaluate(x, c7, c6, c5, c4, c3, c2, c1, c0));
      }

      inline virtual T operator() (const T& x, const T& c8, const T& c7, const T& c6, const T& c5,
                                   const T& c4, const T& c3, const T& c2, const T& c1, const T& c0)
      {
         poly_rtrn(8) (poly_impl<T,8>::evaluate(x, c8, c7, c6, c5, c4, c3, c2, c1, c0));
      }

      inline virtual T operator() (const T& x, const T& c9, const T& c8, const T& c7, const T& c6,
                                   const T& c5, const T& c4, const T& c3, const T& c2, const T& c1,
                                   const T& c0)
      {
         poly_rtrn(9) (poly_impl<T,9>::evaluate(x, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
      }

      inline virtual T operator() (const T& x, const T& c10, const T& c9, const T& c8, const T& c7,
                                   const T& c6, const T& c5, const T& c4, const T& c3, const T& c2,
                                   const T& c1, const T& c0)
      {
         poly_rtrn(10) (poly_impl<T,10>::evaluate(x, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
      }

      inline virtual T operator() (const T& x, const T& c11, const T& c10, const T& c9, const T& c8,
                                   const T& c7, const T& c6, const T& c5, const T& c4, const T& c3,
                                   const T& c2, const T& c1, const T& c0)
      {
         poly_rtrn(11) (poly_impl<T,11>::evaluate(x, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
      }

      inline virtual T operator() (const T& x, const T& c12, const T& c11, const T& c10, const T& c9,
                                   const T& c8, const T& c7, const T& c6, const T& c5, const T& c4,
                                   const T& c3, const T& c2, const T& c1, const T& c0)
      {
         poly_rtrn(12) (poly_impl<T,12>::evaluate(x, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
      }

      #undef poly_rtrn

      inline virtual T operator() ()
      {
         return std::numeric_limits<T>::quiet_NaN();
      }

      inline virtual T operator() (const T&)
      {
         return std::numeric_limits<T>::quiet_NaN();
      }

      inline virtual T operator() (const T&, const T&)
      {
         return std::numeric_limits<T>::quiet_NaN();
      }
   };

   template <typename T>
   class function_compositor
   {
   public:

      typedef exprtk::expression<T>             expression_t;
      typedef exprtk::symbol_table<T>           symbol_table_t;
      typedef exprtk::parser<T>                 parser_t;
      typedef typename parser_t::settings_store settings_t;

      struct function
      {
         function()
         {}

         function(const std::string& n)
         : name_(n)
         {}

         function(const std::string& name,
                  const std::string& expression)
         : name_(name)
         , expression_(expression)
         {}

         function(const std::string& name,
                  const std::string& expression,
                  const std::string& v0)
         : name_(name)
         , expression_(expression)
         {
            v_.push_back(v0);
         }

         function(const std::string& name,
                  const std::string& expression,
                  const std::string& v0, const std::string& v1)
         : name_(name)
         , expression_(expression)
         {
            v_.push_back(v0); v_.push_back(v1);
         }

         function(const std::string& name,
                  const std::string& expression,
                  const std::string& v0, const std::string& v1,
                  const std::string& v2)
         : name_(name)
         , expression_(expression)
         {
            v_.push_back(v0); v_.push_back(v1);
            v_.push_back(v2);
         }

         function(const std::string& name,
                  const std::string& expression,
                  const std::string& v0, const std::string& v1,
                  const std::string& v2, const std::string& v3)
         : name_(name)
         , expression_(expression)
         {
            v_.push_back(v0); v_.push_back(v1);
            v_.push_back(v2); v_.push_back(v3);
         }

         function(const std::string& name,
                  const std::string& expression,
                  const std::string& v0, const std::string& v1,
                  const std::string& v2, const std::string& v3,
                  const std::string& v4)
         : name_(name)
         , expression_(expression)
         {
            v_.push_back(v0); v_.push_back(v1);
            v_.push_back(v2); v_.push_back(v3);
            v_.push_back(v4);
         }

         inline function& name(const std::string& n)
         {
            name_ = n;
            return (*this);
         }

         inline function& expression(const std::string& e)
         {
            expression_ = e;
            return (*this);
         }

         inline function& var(const std::string& v)
         {
            v_.push_back(v);
            return (*this);
         }

         std::string name_;
         std::string expression_;
         std::deque<std::string> v_;
      };

   private:

      struct base_func : public exprtk::ifunction<T>
      {
         typedef const T&                       type;
         typedef exprtk::ifunction<T>     function_t;
         typedef std::vector<T*>            varref_t;
         typedef std::vector<T>                var_t;
         typedef std::pair<T*,std::size_t> lvarref_t;
         typedef std::vector<lvarref_t>    lvr_vec_t;

         using exprtk::ifunction<T>::operator();

         base_func(const std::size_t& pc = 0)
         : exprtk::ifunction<T>(pc)
         , local_var_stack_size(0)
         , stack_depth(0)
         {
            v.resize(pc);
         }

         virtual ~base_func() {}

         #define exprtk_assign(Index)   \
         (*v[Index]) = v##Index; \

         inline void update(const T& v0)
         {
            exprtk_assign(0)
         }

         inline void update(const T& v0, const T& v1)
         {
            exprtk_assign(0) exprtk_assign(1)
         }

         inline void update(const T& v0, const T& v1, const T& v2)
         {
            exprtk_assign(0) exprtk_assign(1)
            exprtk_assign(2)
         }

         inline void update(const T& v0, const T& v1, const T& v2, const T& v3)
         {
            exprtk_assign(0) exprtk_assign(1)
            exprtk_assign(2) exprtk_assign(3)
         }

         inline void update(const T& v0, const T& v1, const T& v2, const T& v3, const T& v4)
         {
            exprtk_assign(0) exprtk_assign(1)
            exprtk_assign(2) exprtk_assign(3)
            exprtk_assign(4)
         }

         inline void update(const T& v0, const T& v1, const T& v2, const T& v3, const T& v4, const T& v5)
         {
            exprtk_assign(0) exprtk_assign(1)
            exprtk_assign(2) exprtk_assign(3)
            exprtk_assign(4) exprtk_assign(5)
         }

         #ifdef exprtk_assign
         #undef exprtk_assign
         #endif

         inline function_t& setup(expression_t& expr)
         {
            expression = expr;

            typedef typename expression_t::control_block::local_data_list_t ldl_t;

            const ldl_t ldl = expr.local_data_list();

            std::vector<std::size_t> index_list;

            for (std::size_t i = 0; i < ldl.size(); ++i)
            {
               if (ldl[i].size)
               {
                  index_list.push_back(i);
               }
            }

            std::size_t input_param_count = 0;

            for (std::size_t i = 0; i < index_list.size(); ++i)
            {
               const std::size_t index = index_list[i];

               if (i < (index_list.size() - v.size()))
               {
                  lv.push_back(
                        std::make_pair(
                           reinterpret_cast<T*>(ldl[index].pointer),
                           ldl[index].size));

                  local_var_stack_size += ldl[index].size;
               }
               else
                  v[input_param_count++] = reinterpret_cast<T*>(ldl[index].pointer);
            }

            clear_stack();

            return (*this);
         }

         inline void pre()
         {
            if (stack_depth++)
            {
               if (!v.empty())
               {
                  var_t var_stack(v.size(),T(0));
                  copy(v,var_stack);
                  param_stack.push_back(var_stack);
               }

               if (!lv.empty())
               {
                  var_t local_var_stack(local_var_stack_size,T(0));
                  copy(lv,local_var_stack);
                  local_stack.push_back(local_var_stack);
               }
            }
         }

         inline void post()
         {
            if (--stack_depth)
            {
               if (!v.empty())
               {
                  copy(param_stack.back(),v);
                  param_stack.pop_back();
               }

               if (!lv.empty())
               {
                  copy(local_stack.back(),lv);
                  local_stack.pop_back();
               }
            }
         }

         void copy(const varref_t& src_v, var_t& dest_v)
         {
            for (std::size_t i = 0; i < src_v.size(); ++i)
            {
               dest_v[i] = (*src_v[i]);
            }
         }

         void copy(const var_t& src_v, varref_t& dest_v)
         {
            for (std::size_t i = 0; i < src_v.size(); ++i)
            {
               (*dest_v[i]) = src_v[i];
            }
         }

         void copy(const lvr_vec_t& src_v, var_t& dest_v)
         {
            typename var_t::iterator itr = dest_v.begin();
            typedef  typename std::iterator_traits<typename var_t::iterator>::difference_type diff_t;

            for (std::size_t i = 0; i < src_v.size(); ++i)
            {
               lvarref_t vr = src_v[i];

               if (1 == vr.second)
                  *itr++ = (*vr.first);
               else
               {
                  std::copy(vr.first, vr.first + vr.second, itr);
                  itr += static_cast<diff_t>(vr.second);
               }
            }
         }

         void copy(const var_t& src_v, lvr_vec_t& dest_v)
         {
            typename var_t::const_iterator itr = src_v.begin();
            typedef  typename std::iterator_traits<typename var_t::iterator>::difference_type diff_t;

            for (std::size_t i = 0; i < src_v.size(); ++i)
            {
               lvarref_t vr = dest_v[i];

               if (1 == vr.second)
                  (*vr.first) = *itr++;
               else
               {
                  std::copy(itr, itr + static_cast<diff_t>(vr.second), vr.first);
                  itr += static_cast<diff_t>(vr.second);
               }
            }
         }

         inline void clear_stack()
         {
            for (std::size_t i = 0; i < v.size(); ++i)
            {
               (*v[i]) = 0;
            }
         }

         inline virtual T value(expression_t& e)
         {
            return e.value();
         }

         expression_t expression;
         varref_t v;
         lvr_vec_t lv;
         std::size_t local_var_stack_size;
         std::size_t stack_depth;
         std::deque<var_t> param_stack;
         std::deque<var_t> local_stack;
      };

      typedef std::map<std::string,base_func*> funcparam_t;

      struct func_0param : public base_func
      {
         using exprtk::ifunction<T>::operator();

         func_0param() : base_func(0) {}

         inline T operator() ()
         {
            return this->value(base_func::expression);
         }
      };

      typedef const T& type;

      template <typename BaseFuncType>
      struct scoped_bft
      {
         explicit scoped_bft(BaseFuncType& bft)
         : bft_(bft)
         {
            bft_.pre ();
         }

        ~scoped_bft()
         {
            bft_.post();
         }

         BaseFuncType& bft_;

      private:

         scoped_bft(const scoped_bft&) exprtk_delete;
         scoped_bft& operator=(const scoped_bft&) exprtk_delete;
      };

      struct func_1param : public base_func
      {
         using exprtk::ifunction<T>::operator();

         func_1param() : base_func(1) {}

         inline T operator() (type v0)
         {
            scoped_bft<func_1param> sb(*this);
            base_func::update(v0);
            return this->value(base_func::expression);
         }
      };

      struct func_2param : public base_func
      {
         using exprtk::ifunction<T>::operator();

         func_2param() : base_func(2) {}

         inline T operator() (type v0, type v1)
         {
            scoped_bft<func_2param> sb(*this);
            base_func::update(v0, v1);
            return this->value(base_func::expression);
         }
      };

      struct func_3param : public base_func
      {
         using exprtk::ifunction<T>::operator();

         func_3param() : base_func(3) {}

         inline T operator() (type v0, type v1, type v2)
         {
            scoped_bft<func_3param> sb(*this);
            base_func::update(v0, v1, v2);
            return this->value(base_func::expression);
         }
      };

      struct func_4param : public base_func
      {
         using exprtk::ifunction<T>::operator();

         func_4param() : base_func(4) {}

         inline T operator() (type v0, type v1, type v2, type v3)
         {
            scoped_bft<func_4param> sb(*this);
            base_func::update(v0, v1, v2, v3);
            return this->value(base_func::expression);
         }
      };

      struct func_5param : public base_func
      {
         using exprtk::ifunction<T>::operator();

         func_5param() : base_func(5) {}

         inline T operator() (type v0, type v1, type v2, type v3, type v4)
         {
            scoped_bft<func_5param> sb(*this);
            base_func::update(v0, v1, v2, v3, v4);
            return this->value(base_func::expression);
         }
      };

      struct func_6param : public base_func
      {
         using exprtk::ifunction<T>::operator();

         func_6param() : base_func(6) {}

         inline T operator() (type v0, type v1, type v2, type v3, type v4, type v5)
         {
            scoped_bft<func_6param> sb(*this);
            base_func::update(v0, v1, v2, v3, v4, v5);
            return this->value(base_func::expression);
         }
      };

      static T return_value(expression_t& e)
      {
         typedef exprtk::results_context<T> results_context_t;
         typedef typename results_context_t::type_store_t type_t;
         typedef typename type_t::scalar_view scalar_t;

         const T result = e.value();

         if (e.return_invoked())
         {
            // Due to the post compilation checks, it can be safely
            // assumed that there will be at least one parameter
            // and that the first parameter will always be scalar.
            return scalar_t(e.results()[0])();
         }

         return result;
      }

      #define def_fp_retval(N)                               \
      struct func_##N##param_retval : public func_##N##param \
      {                                                      \
         inline T value(expression_t& e)                     \
         {                                                   \
            return return_value(e);                          \
         }                                                   \
      };                                                     \

      def_fp_retval(0)
      def_fp_retval(1)
      def_fp_retval(2)
      def_fp_retval(3)
      def_fp_retval(4)
      def_fp_retval(5)
      def_fp_retval(6)

      template <typename Allocator,
                template <typename, typename> class Sequence>
      inline bool add(const std::string& name,
                      const std::string& expression,
                      const Sequence<std::string,Allocator>& var_list,
                      const bool override = false)
      {
         const typename std::map<std::string,expression_t>::iterator itr = expr_map_.find(name);

         if (expr_map_.end() != itr)
         {
            if (!override)
            {
               exprtk_debug(("Compositor error(add): function '%s' already defined\n",
                             name.c_str()));

               return false;
            }

            remove(name, var_list.size());
         }

         if (compile_expression(name, expression, var_list))
         {
            const std::size_t n = var_list.size();

            fp_map_[n][name]->setup(expr_map_[name]);

            return true;
         }
         else
         {
            exprtk_debug(("Compositor error(add): Failed to compile function '%s'\n",
                          name.c_str()));

            return false;
         }
      }

   public:

      function_compositor()
      : parser_(settings_t::compile_all_opts +
                settings_t::e_disable_zero_return)
      , fp_map_(7)
      {}

      function_compositor(const symbol_table_t& st)
      : symbol_table_(st)
      , parser_(settings_t::compile_all_opts +
                settings_t::e_disable_zero_return)
      , fp_map_(7)
      {}

     ~function_compositor()
      {
         clear();
      }

      inline symbol_table_t& symbol_table()
      {
         return symbol_table_;
      }

      inline const symbol_table_t& symbol_table() const
      {
         return symbol_table_;
      }

      inline void add_auxiliary_symtab(symbol_table_t& symtab)
      {
         auxiliary_symtab_list_.push_back(&symtab);
      }

      void clear()
      {
         symbol_table_.clear();
         expr_map_    .clear();

         for (std::size_t i = 0; i < fp_map_.size(); ++i)
         {
            typename funcparam_t::iterator itr = fp_map_[i].begin();
            typename funcparam_t::iterator end = fp_map_[i].end  ();

            while (itr != end)
            {
               delete itr->second;
               ++itr;
            }

            fp_map_[i].clear();
         }
      }

      inline bool add(const function& f, const bool override = false)
      {
         return add(f.name_, f.expression_, f.v_,override);
      }

   private:

      template <typename Allocator,
                template <typename, typename> class Sequence>
      bool compile_expression(const std::string& name,
                              const std::string& expression,
                              const Sequence<std::string,Allocator>& input_var_list,
                              bool  return_present = false)
      {
         expression_t compiled_expression;
         symbol_table_t local_symbol_table;

         local_symbol_table.load_from(symbol_table_);
         local_symbol_table.add_constants();

         if (!valid(name,input_var_list.size()))
            return false;

         if (!forward(name,
                      input_var_list.size(),
                      local_symbol_table,
                      return_present))
            return false;

         compiled_expression.register_symbol_table(local_symbol_table);

         for (std::size_t i = 0; i < auxiliary_symtab_list_.size(); ++i)
         {
            compiled_expression.register_symbol_table((*auxiliary_symtab_list_[i]));
         }

         std::string mod_expression;

         for (std::size_t i = 0; i < input_var_list.size(); ++i)
         {
            mod_expression += " var " + input_var_list[i] + "{};\n";
         }

         if (
              ('{' == details::front(expression)) &&
              ('}' == details::back (expression))
            )
            mod_expression += "~" + expression + ";";
         else
            mod_expression += "~{" + expression + "};";

         if (!parser_.compile(mod_expression,compiled_expression))
         {
            exprtk_debug(("Compositor Error: %s\n",parser_.error().c_str()));
            exprtk_debug(("Compositor modified expression: \n%s\n",mod_expression.c_str()));

            remove(name,input_var_list.size());

            return false;
         }

         if (!return_present && parser_.dec().return_present())
         {
            remove(name,input_var_list.size());

            return compile_expression(name, expression, input_var_list, true);
         }

         // Make sure every return point has a scalar as its first parameter
         if (parser_.dec().return_present())
         {
            typedef std::vector<std::string> str_list_t;

            str_list_t ret_param_list = parser_.dec().return_param_type_list();

            for (std::size_t i = 0; i < ret_param_list.size(); ++i)
            {
               const std::string& params = ret_param_list[i];

               if (params.empty() || ('T' != params[0]))
               {
                  exprtk_debug(("Compositor Error: Return statement in function '%s' is invalid\n",
                                name.c_str()));

                  remove(name,input_var_list.size());

                  return false;
               }
            }
         }

         expr_map_[name] = compiled_expression;

         exprtk::ifunction<T>& ifunc = (*(fp_map_[input_var_list.size()])[name]);

         if (symbol_table_.add_function(name,ifunc))
            return true;
         else
         {
            exprtk_debug(("Compositor Error: Failed to add function '%s' to symbol table\n",
                          name.c_str()));
            return false;
         }
      }

      inline bool symbol_used(const std::string& symbol) const
      {
         return (
                  symbol_table_.is_variable       (symbol) ||
                  symbol_table_.is_stringvar      (symbol) ||
                  symbol_table_.is_function       (symbol) ||
                  symbol_table_.is_vector         (symbol) ||
                  symbol_table_.is_vararg_function(symbol)
                );
      }

      inline bool valid(const std::string& name,
                        const std::size_t& arg_count) const
      {
         if (arg_count > 6)
            return false;
         else if (symbol_used(name))
            return false;
         else if (fp_map_[arg_count].end() != fp_map_[arg_count].find(name))
            return false;
         else
            return true;
      }

      inline bool forward(const std::string& name,
                          const std::size_t& arg_count,
                          symbol_table_t& sym_table,
                          const bool ret_present = false)
      {
         switch (arg_count)
         {
            #define case_stmt(N)                                     \
            case N : (fp_map_[arg_count])[name] =                    \
                     (!ret_present) ? static_cast<base_func*>        \
                                      (new func_##N##param) :        \
                                      static_cast<base_func*>        \
                                      (new func_##N##param_retval) ; \
                     break;                                          \

            case_stmt(0) case_stmt(1) case_stmt(2)
            case_stmt(3) case_stmt(4) case_stmt(5)
            case_stmt(6)
            #undef case_stmt
         }

         exprtk::ifunction<T>& ifunc = (*(fp_map_[arg_count])[name]);

         return sym_table.add_function(name,ifunc);
      }

      inline void remove(const std::string& name, const std::size_t& arg_count)
      {
         if (arg_count > 6)
            return;

         const typename std::map<std::string,expression_t>::iterator em_itr = expr_map_.find(name);

         if (expr_map_.end() != em_itr)
         {
            expr_map_.erase(em_itr);
         }

         const typename funcparam_t::iterator fp_itr = fp_map_[arg_count].find(name);

         if (fp_map_[arg_count].end() != fp_itr)
         {
            delete fp_itr->second;
            fp_map_[arg_count].erase(fp_itr);
         }

         symbol_table_.remove_function(name);
      }

   private:

      symbol_table_t symbol_table_;
      parser_t parser_;
      std::map<std::string,expression_t> expr_map_;
      std::vector<funcparam_t> fp_map_;
      std::vector<symbol_table_t*> auxiliary_symtab_list_;
   }; // class function_compositor

   template <typename T>
   inline bool pgo_primer()
   {
      static const std::string expression_list[] =
             {
                "(y + x)",
                "2 * (y + x)",
                "(2 * y + 2 * x)",
                "(y + x / y) * (x - y / x)",
                "x / ((x + y) * (x - y)) / y",
                "1 - ((x * y) + (y / x)) - 3",
                "sin(2 * x) + cos(pi / y)",
                "1 - sin(2 * x) + cos(pi / y)",
                "sqrt(1 - sin(2 * x) + cos(pi / y) / 3)",
                "(x^2 / sin(2 * pi / y)) -x / 2",
                "x + (cos(y - sin(2 / x * pi)) - sin(x - cos(2 * y / pi))) - y",
                "clamp(-1.0, sin(2 * pi * x) + cos(y / 2 * pi), +1.0)",
                "iclamp(-1.0, sin(2 * pi * x) + cos(y / 2 * pi), +1.0)",
                "max(3.33, min(sqrt(1 - sin(2 * x) + cos(pi / y) / 3), 1.11))",
                "if(avg(x,y) <= x + y, x - y, x * y) + 2 * pi / x",
                "1.1x^1 + 2.2y^2 - 3.3x^3 + 4.4y^4 - 5.5x^5 + 6.6y^6 - 7.7x^27 + 8.8y^55",
                "(yy + xx)",
                "2 * (yy + xx)",
                "(2 * yy + 2 * xx)",
                "(yy + xx / yy) * (xx - yy / xx)",
                "xx / ((xx + yy) * (xx - yy)) / yy",
                "1 - ((xx * yy) + (yy / xx)) - 3",
                "sin(2 * xx) + cos(pi / yy)",
                "1 - sin(2 * xx) + cos(pi / yy)",
                "sqrt(1 - sin(2 * xx) + cos(pi / yy) / 3)",
                "(xx^2 / sin(2 * pi / yy)) -xx / 2",
                "xx + (cos(yy - sin(2 / xx * pi)) - sin(xx - cos(2 * yy / pi))) - yy",
                "clamp(-1.0, sin(2 * pi * xx) + cos(yy / 2 * pi), +1.0)",
                "max(3.33, min(sqrt(1 - sin(2 * xx) + cos(pi / yy) / 3), 1.11))",
                "if(avg(xx,yy) <= xx + yy, xx - yy, xx * yy) + 2 * pi / xx",
                "1.1xx^1 + 2.2yy^2 - 3.3xx^3 + 4.4yy^4 - 5.5xx^5 + 6.6yy^6 - 7.7xx^27 + 8.8yy^55",
                "(1.1*(2.2*(3.3*(4.4*(5.5*(6.6*(7.7*(8.8*(9.9+x)))))))))",
                "(((((((((x+9.9)*8.8)*7.7)*6.6)*5.5)*4.4)*3.3)*2.2)*1.1)",
                "(x + y) * z", "x + (y * z)", "(x + y) * 7", "x + (y * 7)",
                "(x + 7) * y", "x + (7 * y)", "(7 + x) * y", "7 + (x * y)",
                "(2 + x) * 3", "2 + (x * 3)", "(2 + 3) * x", "2 + (3 * x)",
                "(x + 2) * 3", "x + (2 * 3)",
                "(x + y) * (z / w)", "(x + y) * (z / 7)", "(x + y) * (7 / z)", "(x + 7) * (y / z)",
                "(7 + x) * (y / z)", "(2 + x) * (y / z)", "(x + 2) * (y / 3)", "(2 + x) * (y / 3)",
                "(x + 2) * (3 / y)", "x + (y * (z / w))", "x + (y * (z / 7))", "x + (y * (7 / z))",
                "x + (7 * (y / z))", "7 + (x * (y / z))", "2 + (x * (3 / y))", "x + (2 * (y / 4))",
                "2 + (x * (y / 3))", "x + (2 * (3 / y))",
                "x + ((y * z) / w)", "x + ((y * z) / 7)", "x + ((y * 7) / z)", "x + ((7 * y) / z)",
                "7 + ((y * z) / w)", "2 + ((x * 3) / y)", "x + ((2 * y) / 3)", "2 + ((x * y) / 3)",
                "x + ((2 * 3) / y)", "(((x + y) * z) / w)",
                "(((x + y) * z) / 7)", "(((x + y) * 7) / z)", "(((x + 7) * y) / z)", "(((7 + x) * y) / z)",
                "(((2 + x) * 3) / y)", "(((x + 2) * y) / 3)", "(((2 + x) * y) / 3)", "(((x + 2) * 3) / y)",
                "((x + (y * z)) / w)", "((x + (y * z)) / 7)", "((x + (y * 7)) / y)", "((x + (7 * y)) / z)",
                "((7 + (x * y)) / z)", "((2 + (x * 3)) / y)", "((x + (2 * y)) / 3)", "((2 + (x * y)) / 3)",
                "((x + (2 * 3)) / y)",
                "(xx + yy) * zz", "xx + (yy * zz)",
                "(xx + yy) * 7", "xx + (yy * 7)",
                "(xx + 7) * yy", "xx + (7 * yy)",
                "(7 + xx) * yy", "7 + (xx * yy)",
                "(2 + x) * 3", "2 + (x * 3)",
                "(2 + 3) * x", "2 + (3 * x)",
                "(x + 2) * 3", "x + (2 * 3)",
                "(xx + yy) * (zz / ww)", "(xx + yy) * (zz / 7)",
                "(xx + yy) * (7 / zz)", "(xx + 7) * (yy / zz)",
                "(7 + xx) * (yy / zz)", "(2 + xx) * (yy / zz)",
                "(xx + 2) * (yy / 3)", "(2 + xx) * (yy / 3)",
                "(xx + 2) * (3 / yy)", "xx + (yy * (zz / ww))",
                "xx + (yy * (zz / 7))", "xx + (yy * (7 / zz))",
                "xx + (7 * (yy / zz))", "7 + (xx * (yy / zz))",
                "2 + (xx * (3 / yy))", "xx + (2 * (yy / 4))",
                "2 + (xx * (yy / 3))", "xx + (2 * (3 / yy))",
                "xx + ((yy * zz) / ww)", "xx + ((yy * zz) / 7)",
                "xx + ((yy * 7) / zz)", "xx + ((7 * yy) / zz)",
                "7 + ((yy * zz) / ww)", "2 + ((xx * 3) / yy)",
                "xx + ((2 * yy) / 3)", "2 + ((xx * yy) / 3)",
                "xx + ((2 * 3) / yy)", "(((xx + yy) * zz) / ww)",
                "(((xx + yy) * zz) / 7)", "(((xx + yy) * 7) / zz)",
                "(((xx + 7) * yy) / zz)", "(((7 + xx) * yy) / zz)",
                "(((2 + xx) * 3) / yy)", "(((xx + 2) * yy) / 3)",
                "(((2 + xx) * yy) / 3)", "(((xx + 2) * 3) / yy)",
                "((xx + (yy * zz)) / ww)", "((xx + (yy * zz)) / 7)",
                "((xx + (yy * 7)) / yy)", "((xx + (7 * yy)) / zz)",
                "((7 + (xx * yy)) / zz)", "((2 + (xx * 3)) / yy)",
                "((xx + (2 * yy)) / 3)", "((2 + (xx * yy)) / 3)",
                "((xx + (2 * 3)) / yy)"
             };

      static const std::size_t expression_list_size = sizeof(expression_list) / sizeof(std::string);

      T  x = T(0);
      T  y = T(0);
      T  z = T(0);
      T  w = T(0);
      T xx = T(0);
      T yy = T(0);
      T zz = T(0);
      T ww = T(0);

      exprtk::symbol_table<T> symbol_table;
      symbol_table.add_constants();
      symbol_table.add_variable( "x", x);
      symbol_table.add_variable( "y", y);
      symbol_table.add_variable( "z", z);
      symbol_table.add_variable( "w", w);
      symbol_table.add_variable("xx",xx);
      symbol_table.add_variable("yy",yy);
      symbol_table.add_variable("zz",zz);
      symbol_table.add_variable("ww",ww);

      typedef typename std::deque<exprtk::expression<T> > expr_list_t;
      expr_list_t expr_list;

      const std::size_t rounds = 50;

      {
         for (std::size_t r = 0; r < rounds; ++r)
         {
            expr_list.clear();
            exprtk::parser<T> parser;

            for (std::size_t i = 0; i < expression_list_size; ++i)
            {
               exprtk::expression<T> expression;
               expression.register_symbol_table(symbol_table);

               if (!parser.compile(expression_list[i],expression))
               {
                  return false;
               }

               expr_list.push_back(expression);
            }
         }
      }

      struct execute
      {
         static inline T process(T& x, T& y, expression<T>& expression)
         {
            static const T lower_bound = T(-20);
            static const T upper_bound = T(+20);
            static const T delta       = T(0.1);

            T total = T(0);

            for (x = lower_bound; x <= upper_bound; x += delta)
            {
               for (y = lower_bound; y <= upper_bound; y += delta)
               {
                  total += expression.value();
               }
            }

            return total;
         }
      };

      for (std::size_t i = 0; i < expr_list.size(); ++i)
      {
         execute::process( x,  y, expr_list[i]);
         execute::process(xx, yy, expr_list[i]);
      }

      {
         for (std::size_t i = 0; i < 10000; ++i)
         {
            const T v = T(123.456 + i);

            if (details::is_true(details::numeric::nequal(details::numeric::fast_exp<T, 1>::result(v),details::numeric::pow(v,T(1)))))
               return false;

            #define else_stmt(N)                                                                                                           \
            else if (details::is_true(details::numeric::nequal(details::numeric::fast_exp<T,N>::result(v),details::numeric::pow(v,T(N))))) \
               return false;                                                                                                               \

            else_stmt( 2) else_stmt( 3) else_stmt( 4) else_stmt( 5)
            else_stmt( 6) else_stmt( 7) else_stmt( 8) else_stmt( 9)
            else_stmt(10) else_stmt(11) else_stmt(12) else_stmt(13)
            else_stmt(14) else_stmt(15) else_stmt(16) else_stmt(17)
            else_stmt(18) else_stmt(19) else_stmt(20) else_stmt(21)
            else_stmt(22) else_stmt(23) else_stmt(24) else_stmt(25)
            else_stmt(26) else_stmt(27) else_stmt(28) else_stmt(29)
            else_stmt(30) else_stmt(31) else_stmt(32) else_stmt(33)
            else_stmt(34) else_stmt(35) else_stmt(36) else_stmt(37)
            else_stmt(38) else_stmt(39) else_stmt(40) else_stmt(41)
            else_stmt(42) else_stmt(43) else_stmt(44) else_stmt(45)
            else_stmt(46) else_stmt(47) else_stmt(48) else_stmt(49)
            else_stmt(50) else_stmt(51) else_stmt(52) else_stmt(53)
            else_stmt(54) else_stmt(55) else_stmt(56) else_stmt(57)
            else_stmt(58) else_stmt(59) else_stmt(60) else_stmt(61)
         }
      }

      return true;
   }
}

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#   ifndef NOMINMAX
#      define NOMINMAX
#   endif
#   ifndef WIN32_LEAN_AND_MEAN
#      define WIN32_LEAN_AND_MEAN
#   endif
#   include <windows.h>
#   include <ctime>
#else
#   include <ctime>
#   include <sys/time.h>
#   include <sys/types.h>
#endif

namespace exprtk
{
   class timer
   {
   public:

      #if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
      timer()
      : in_use_(false)
      {
         QueryPerformanceFrequency(&clock_frequency_);
      }

      inline void start()
      {
         in_use_ = true;
         QueryPerformanceCounter(&start_time_);
      }

      inline void stop()
      {
         QueryPerformanceCounter(&stop_time_);
         in_use_ = false;
      }

      inline double time() const
      {
         return (1.0 * (stop_time_.QuadPart - start_time_.QuadPart)) / (1.0 * clock_frequency_.QuadPart);
      }

      #else

      timer()
      : in_use_(false)
      {
         start_time_.tv_sec  = 0;
         start_time_.tv_usec = 0;

         stop_time_.tv_sec   = 0;
         stop_time_.tv_usec  = 0;
      }

      inline void start()
      {
         in_use_ = true;
         gettimeofday(&start_time_,0);
      }

      inline void stop()
      {
         gettimeofday(&stop_time_, 0);
         in_use_ = false;
      }

      inline unsigned long long int usec_time() const
      {
         if (!in_use_)
         {
            if (stop_time_.tv_sec >= start_time_.tv_sec)
            {
               return 1000000LLU * static_cast<details::_uint64_t>(stop_time_.tv_sec  - start_time_.tv_sec ) +
                                   static_cast<details::_uint64_t>(stop_time_.tv_usec - start_time_.tv_usec) ;
            }
            else
               return std::numeric_limits<details::_uint64_t>::max();
         }
         else
            return std::numeric_limits<details::_uint64_t>::max();
      }

      inline double time() const
      {
         return usec_time() * 0.000001;
      }

      #endif

      inline bool in_use() const
      {
         return in_use_;
      }

   private:

      bool in_use_;

      #if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
         LARGE_INTEGER start_time_;
         LARGE_INTEGER stop_time_;
         LARGE_INTEGER clock_frequency_;
      #else
         struct timeval start_time_;
         struct timeval stop_time_;
      #endif
   };

   template <typename T>
   struct type_defs
   {
      typedef symbol_table<T>         symbol_table_t;
      typedef expression<T>           expression_t;
      typedef parser<T>               parser_t;
      typedef parser_error::type      error_t;
      typedef function_compositor<T>  compositor_t;
      typedef typename compositor_t::function function_t;
   };

} // namespace exprtk

namespace exprtk
{
   namespace information
   {
      static const char* library = "Mathematical Expression Toolkit";
      static const char* version = "2.7182818284590452353602874713526"
                                   "624977572470936999595749669676277"
                                   "240766303535475945713821785251664"
                                   "274274663919320030599218174135966";
      static const char* date    = "20220101";

      static inline std::string data()
      {
         static const std::string info_str = std::string(library) +
                                             std::string(" v") + std::string(version) +
                                             std::string(" (") + date + std::string(")");
         return info_str;
      }

   } // namespace information

   #ifdef exprtk_debug
   #undef exprtk_debug
   #endif

   #ifdef exprtk_error_location
   #undef exprtk_error_location
   #endif

   #ifdef exprtk_disable_fallthrough_begin
   #undef exprtk_disable_fallthrough_begin
   #endif

   #ifdef exprtk_disable_fallthrough_end
   #undef exprtk_disable_fallthrough_end
   #endif

   #ifdef exprtk_override
   #undef exprtk_override
   #endif

   #ifdef exprtk_final
   #undef exprtk_final
   #endif

   #ifdef exprtk_delete
   #undef exprtk_delete
   #endif

} // namespace exprtk

#endif
