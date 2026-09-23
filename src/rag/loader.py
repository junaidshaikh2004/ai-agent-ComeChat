from langchain_community.document_loaders import DirectoryLoader,TextLoader,ToMarkdownLoader
import frontmatter

def directory_loader(): 
    loader = DirectoryLoader(
      path = "knowledge-base",
      glob="**/*.md",
      loader_cls=TextLoader,
      loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    
    for doc in documents :
        post = frontmatter.loads(doc.page_content)
        doc.metadata.update(post.metadata)
        doc.page_content = post.content
                
    
    print(f"Loaded {len(documents)} documents \n ")    
    return documents







