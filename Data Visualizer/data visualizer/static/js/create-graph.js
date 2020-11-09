/*
 * Parse the data and create a graph with the data.
 */
function parseData(createGraph) {
    Papa.parse("static/data/spanish-silver.csv", {
        download: true,
        complete: function(results) {
            createGraph(results.data);
        }
    });
}

function createGraph(data) {
    let date="year";
    let xParam="silver_minted";
    let YParam="situados_paid";
    let time= "time";
    var parx1=[];
    var pary1=[];
    var pary2=[];
    var parTime=[]
    var a;
    var b;
    var d;
    for(var i=0;i<data[0].length;++i){
        if(data[0][i]==date)
            a=i;
        else if(data[0][i]==xParam)
            b=i;
        else if(data[0][i]==YParam)
            d=i;
        else 
            e=i;
    }
    for (var i = 1; i < data.length; i++) {
        pary1.push(Number(data[i][b]));
        parx1.push(Number(data[i][a]));
        pary2.push(Number(data[i][d]));
        parTime.push(Number(data[i][e]))
    }


    //if you want the categories only
    // drawPie(pary1,[],0,1,parx1,parTime,1720,0);

    //if you want the categories and any list only
    drawPie(pary1,pary1,0,1,parx1,parTime,1720,0);

    //parX1 in drawLine represents the categories likee others. so, the numbers on X-axis will be pary1 don't forget
    //make it like this to make it similar in all drawings
    // drawLine(parx1,pary1);

    //if you want the categories only (filter by date and time)
    // drawBar(pary1,[],0,1,parx1,parTime,1720,0);

    //if you want the categories and any list only (filter by date and time)
    // drawBar(pary1,pary1,0,1,parx1,parTime,1720,0);

    //if you want the categories and any list only (filter by date and time)
    // drawBar(pary1,pary1,0,1,parx1,parTime,1720,0);

    //if you want the categories and any list only 
    // drawBar(pary1,pary1,0);

    //if you want the categories and any list only 
    // drawPie(pary1,pary1,0);

}

parseData(createGraph);



//the freq if you don't put paramY1 is the freq of categories in paramX1, but if you 
//put it, the freq is the sum of all values in paramY1 which refers to its category in paramX1
function drawPie(paramX1,paramY1,ispercented, filter=0 , date=[], time=[] , dateNeeded=null, timeNeeded=null) {

    if(filter==0){ //not filtering by date and time, will call endpoint calc_freq

        var dict={
            "paramX1":paramX1,
            "paramY1":paramY1
        };
        $.ajax({
            type : "POST",
            url : '/calc_freq',
            dataType: "json",
            data: JSON.stringify(dict),
            contentType: 'application/json;charset=UTF-8',
            success: function (data) {
                cat=data["unique"];
                freq=data["freq"];
                sum=data["sum"];

                let ddd = [];
                for (let i = 0; i < cat.length; ++i) {
                    if(ispercented)
                        ddd.push(['cat' + String(cat[i]), freq[i]/sum]);
                    else
                        ddd.push(['cat' + String(cat[i]), freq[i]]);
                }
                var chart = c3.generate({
                    data: {
                        columns: ddd,
                        type: 'pie',
                    
                        onclick: function(d, i) { console.log("onclick", d, i); },
                        onmouseover: function(d, i) { console.log("onmouseover", d, i); },
                        onmouseout: function(d, i) { console.log("onmouseout", d, i); }
                    }
                });
            }
        });
    }
    else{
            //if filtering by date and time, will call endpoint calc_freq_date_time
        var dict={
            "paramX1":paramX1,
            "paramY1":paramY1,
            "date": date,
            "time": time,
            "dateNeeded":dateNeeded,
            "timeNeeded":timeNeeded
        };
        $.ajax({
            type : "POST",
            url : '/calc_freq_date_time',
            dataType: "json",
            data: JSON.stringify(dict),
            contentType: 'application/json;charset=UTF-8',
            success: function (data) {
                cat=data["unique"];
                freq=data["freq"];
                sum=data["sum"];

                let ddd = [];
                for (let i = 0; i < cat.length; ++i) {
                    if(ispercented)
                        ddd.push(['cat' + String(cat[i]), freq[i]/sum]);
                    else
                        ddd.push(['cat' + String(cat[i]), freq[i]]);
                }
                var chart = c3.generate({
                    data: {
                        columns: ddd,
                        type: 'pie',
                    
                        onclick: function(d, i) { console.log("onclick", d, i); },
                        onmouseover: function(d, i) { console.log("onmouseover", d, i); },
                        onmouseout: function(d, i) { console.log("onmouseout", d, i); }
                    }
                });
            }
        });

    }
}

function drawLine(paramX1,paramY1) {

    //will call calc_freq_for_line endpoint
    var dict={
        "paramX1":paramX1,
        "paramY1":paramY1
    };
    $.ajax({
        type : "POST",
        url : '/calc_freq_for_line',
        dataType: "json",
        data: JSON.stringify(dict),
        contentType: 'application/json;charset=UTF-8',
        success: function (data) {
            cat=data["unique"];
            freq=data["freq"];
            let X= paramY1;
            X.unshift("x");
            let ddd = [];
            ddd.push(X);
            for (let i = 0; i < cat.length; ++i) {
              freq[i].unshift('cat' + String(cat[i]));
                ddd.push(freq[i]);
            }
            var chart = c3.generate({
                data: {
                    x: 'x',
                    columns: ddd
                }
            });
        }
    });
}

function drawBar(paramX1,paramY1,ispercented, filter=0 , date=[], time=[] , dateNeeded=null, timeNeeded=null) {
    if(filter==0){ //not filtering by date and time, will call endpoint calc_freq
        var dict={
            "paramX1":paramX1,
            "paramY1":paramY1
        };
        $.ajax({
            type : "POST",
            url : '/calc_freq',
            dataType: "json",
            data: JSON.stringify(dict),
            contentType: 'application/json;charset=UTF-8',
            success: function (data) {
                cat=data["unique"];
                freq=data["freq"];
                sum=data["sum"];
                let ddd = [];
                for (let i = 0; i < cat.length; ++i) {
                        ddd.push(['cat' + String(cat[i]), freq[i]]);
                }
                var chart = c3.generate({
                    data: {
                        columns: ddd,
                        type: 'bar',
                        colors: {
                            data1: '#ff0000',
                            data2: '#00ff00',
                            data3: '#0000ff'
                        },
                        color: function(color, d) {
                            // d will be 'id' when called for legends
                            return d.id && d.id === 'data3' ? d3.rgb(color).darker(d.value / 150) : color;
                        }
                    }
                });
                }
            });
    }
    else{
        //if filtering by date and time, will call endpoint calc_freq_date_time
        var dict={
            "paramX1":paramX1,
            "paramY1":paramY1,
            "date": date,
            "time": time,
            "dateNeeded":dateNeeded,
            "timeNeeded":timeNeeded
        };
        $.ajax({
            type : "POST",
            url : '/calc_freq_date_time',
            dataType: "json",
            data: JSON.stringify(dict),
            contentType: 'application/json;charset=UTF-8',
            success: function (data) {
                cat=data["unique"];
                freq=data["freq"];
                sum=data["sum"];

                let ddd = [];
                for (let i = 0; i < cat.length; ++i) {
                    if(ispercented)
                        ddd.push(['cat' + String(cat[i]), freq[i]/sum]);
                    else
                        ddd.push(['cat' + String(cat[i]), freq[i]]);
                }
                var chart = c3.generate({
                    data: {
                        columns: ddd,
                        type: 'bar',
                        colors: {
                            data1: '#ff0000',
                            data2: '#00ff00',
                            data3: '#0000ff'
                        },
                        color: function(color, d) {
                            // d will be 'id' when called for legends
                            return d.id && d.id === 'data3' ? d3.rgb(color).darker(d.value / 150) : color;
                        }
                    }
                });
            }
        });

    }
}

