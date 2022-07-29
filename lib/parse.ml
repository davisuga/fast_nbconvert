let open_file path env = Eio.Dir.load (Eio.Stdenv.cwd env) path

let get_sources_paths ext path =
  Utils.dir_contents path |> List.filter (String.ends_with ~suffix:ext)

let with_cell_counter num cell = Printf.sprintf "# In[%i]\n%s" num cell
let create_file = try Sys.argv.(2) = "--make-file" with _ -> false

let make_py_file path =
  let open Ipynb_types in
  let open Ppx_deriving_yojson_runtime in
  ( Yojson.Safe.from_file path
  |> python_notebook_of_yojson
  >|= (fun notebook ->
        notebook.cells
        |> List.mapi (fun index { cell_type; source } ->
               match cell_type with
               | "code" ->
                   with_cell_counter index
                     (String.concat ""
                        (source
                        |> List.map (fun src ->
                               if Utils.contains src "!pip" then "# " ^ src
                               else src)))
               | "markdown" ->
                   with_cell_counter index
                     (String.concat "" (source |> List.map (fun s -> "# " ^ s)))
               | _ -> ""))
  >|= String.concat "\n"
  >|= fun result ->
    if create_file then
      output_string
        (open_out
           ((path
            |> String.split_on_char '.'
            |> Base.List.drop_last
            |> Option.value ~default:[]
            |> String.concat ".")
           ^ ".py"))
        result
    else print_string result )
  |> function
  | Ok _ -> "nicee"
  | Error e -> failwith ("Fuck " ^ e)
