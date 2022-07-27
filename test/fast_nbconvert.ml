(* Build with `ocamlbuild -pkg alcotest simple.byte` *)

(* A module with functions to test *)
module To_test = struct
  let lowercase = String.lowercase_ascii
  let capitalize = String.capitalize_ascii
  let str_concat = String.concat ""
end

(* The tests *)
let test_lowercase () =
  Alcotest.(check string) "same string" "hello!" (To_test.lowercase "hELLO!")

let test_capitalize () =
  Alcotest.(check string) "same string" "World." (To_test.capitalize "world.")

let test_str_concat () =
  Alcotest.(check string)
    "same string" "foobar"
    (To_test.str_concat [ "foo"; "bar" ])

let test_source_listing () =
  Fast_nbconvert_lib.Parse.get_sources_paths ".ml" "."
  |> List.map (Alcotest.(check string) "Same string" "./fast_nbconvert.ml")
  |> ignore

(* Run it *)
let () =
  let open Alcotest in
  run "Utils"
    [
      ( "string-case",
        [
          test_case "Lower case" `Quick test_lowercase;
          test_case "Capitalization" `Quick test_capitalize;
        ] );
      ("string-concat", [ test_case "String mashing" `Quick test_str_concat ]);
      ("list-concat", [ test_case "List mashing" `Slow test_source_listing ]);
    ]
