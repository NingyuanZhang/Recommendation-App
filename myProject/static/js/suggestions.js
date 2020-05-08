$(document).ready(function(){
  $('ul#menu li').hover(function(){
     $(this).children('ul').finish().slideDown('medium');
  }, function(){
     $(this).children('ul').finish().slideUp('medium');
  });

  $('#cat_1 #cat_sug').on("click", function(e){
    e.stopPropagation();
    e.preventDefault();
    console.log('cat_1 ' + e.target.getAttribute('data-val'))
    // var s1 = e.target.getAttribute('data-val');
    $("input[name=suggestion1]").val(e.target.getAttribute('data-val'));
  });

  $('#cat_2 #cat_sug').on("click", function(e){
    e.stopPropagation();
    e.preventDefault();
    console.log('cat_2 ' + e.target.getAttribute('data-val'))
    // var s1 = e.target.getAttribute('data-val');
    $("input[name=suggestion2]").val(e.target.getAttribute('data-val'));
  }); 

  $('#cat_3 #cat_sug').on("click", function(e){
    e.stopPropagation();
    e.preventDefault();
    console.log('cat_3 ' + e.target.getAttribute('data-val'))
    // var s1 = e.target.getAttribute('data-val');
    $("input[name=suggestion3]").val(e.target.getAttribute('data-val'));
  });

  // $('#cat_A li').on("click", function(e){
  //   e.stopPropagation();
  //   e.preventDefault();
  //   console.log('cat_A ' + e.target.getAttribute('data-val'))
  //   // var s1 = e.target.getAttribute('data-val');
  //   $("input[name=suggestion3]").val(e.target.getAttribute('data-val'));
  // });

  // $('#model2 li').on("click", function(e){
  //   e.stopPropagation();
  //   e.preventDefault();
  //   console.log('model2 ' + e.target.getAttribute('data-val'))
  //   $("input[name=suggestion2]").val(e.target.getAttribute('data-val'));
  //   // var s2 = e.target.getAttribute('data-val');
  //   // $("input[name=suggestion2]").val(s2);
  // });
  // $('#model3 li').on("click", function(e){
  //   e.stopPropagation();
  //   e.preventDefault();
  //   console.log('model3 ' + e.target.getAttribute('data-val'))
  //   $("input[name=suggestion3]").val(e.target.getAttribute('data-val'));
  //   // var s3 = e.target.getAttribute('data-val');
  //   // $("input[name=suggestion3]").val(s3);
  // });

  $("form").submit(function(){
    var s1 = $("input[name=suggestion1]").val();
    var s2 = $("input[name=suggestion2]").val();
    var s3 = $("input[name=suggestion3]").val();
    console.log(s1, s2, s3);
  });
});