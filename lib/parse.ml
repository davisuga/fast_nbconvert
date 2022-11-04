(* let open_file path env = Eio.Dir.load (Eio.Stdenv.cwd env) path *)

let get_sources_paths ext path =
  Utils.dir_contents path |> List.filter (String.ends_with ~suffix:ext)

let with_cell_counter num cell = Printf.sprintf "# In[%i]\n%s" num cell
(* let create_file = try Sys.argv.(2) = "--make-file" with _ -> false *)

let print_py_code_cell source index =
  with_cell_counter index
    (String.concat ""
       (source
       |> List.map (fun src ->
              if Utils.contains src "!pip" then "# " ^ src else src)))

let print_py_md_cell source index =
  with_cell_counter index
    (String.concat "" (source |> List.map (fun s -> "# " ^ s)))

let (print_py_cell : int -> Ipynb_types.cell -> string) =
 fun index { cell_type; source } ->
  match cell_type with
  | "code" -> print_py_code_cell source index
  | "markdown" -> print_py_md_cell source index
  | _ -> ""

let get_py_file_path path =
  (path
  |> String.split_on_char '.'
  |> Base.List.drop_last
  |> Option.value ~default:[]
  |> String.concat ".")
  ^ ".py"

let write_py_files path =
  open_out
    ((path
     |> String.split_on_char '.'
     |> Base.List.drop_last
     |> Option.value ~default:[]
     |> String.concat ".")
    ^ ".py")

let make_py_file ~create_file path =
  let open Ipynb_types in
  let open Ppx_deriving_yojson_runtime in
  ( Yojson.Safe.from_file path
  |> python_notebook_of_yojson
  >|= (fun notebook -> notebook.cells |> List.mapi print_py_cell)
  >|= String.concat "\n"
  >|= fun result ->
    if create_file then output_string (open_out (get_py_file_path path)) result
    else print_string result )
  |> function
  | Ok _ -> "nicee"
  | Error e -> failwith ("Oops " ^ e)
